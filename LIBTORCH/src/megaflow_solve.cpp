// ============================================================================
// megaflow_solve.cpp  —  solve core (part 1: engine load + forwardTrack).
//
// forwardTrack is an op-for-op transcription of MegaFlow.forward_track (the
// origin_T != 2 path, non-training). Validated path: global anchor feature0 =
// features[:,0] computed once over all T; per-window refine; overlap-overwrite
// full_flows[:,ara] (later window wins). All 5 NN stages are engine calls; the
// rest is the transcribed glue (megaflow_glue.cpp).
//
// AUTOCAST CONTRACT: the CALLER sets one outer autocast scope (solveTrajectory
// uses bf16, matching build_full_trajectory; a standalone forwardTrack unit test
// vs reference_solve_window.py sets fp16). Engines are loaded with
// autocastFp16=false so they inherit that scope — exactly like engine_cache.py's
// run_engine (no per-engine autocast). clear_cache() is called once after feature
// extraction to drop the 4.6GB ViT weight-cast cache (e_vit/e_head run once).
//
// solveTrajectory + saveCache are in part 2 (pending megaflow_cache.py).
// ============================================================================
#include "megaflow_solve.h"

#include <torch/nn/functional.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Fn = torch::nn::functional;
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

namespace megaflow {

// ---------------------------------------------------------------------------
// engine loading — 5 .pt from `dir`, IO names matching export_engines.py wiring.
// autocastFp16=false for the solve path (outer scope manages precision).
// ---------------------------------------------------------------------------
Engines loadEngines(const std::string& dir, cudaStream_t /*stream*/, bool autocastFp16)
{
    auto P = [&](const char* n) { return dir + "/" + n + ".pt"; };
    Engines E;
    E.vit    = std::make_unique<TorchEngine>(P("e_vit"), "e_vit",
                   std::vector<std::string>{"img"},
                   std::vector<std::string>{"tokens"}, autocastFp16);
    E.cnn    = std::make_unique<TorchEngine>(P("e_cnn"), "e_cnn",
                   std::vector<std::string>{"frames"},
                   std::vector<std::string>{"cnn0", "cnn1"}, autocastFp16);
    E.head   = std::make_unique<TorchEngine>(P("e_head"), "e_head",
                   std::vector<std::string>{"tokens", "img", "cnn0", "cnn1"},
                   std::vector<std::string>{"feat"}, autocastFp16);
    E.proj   = std::make_unique<TorchEngine>(P("e_proj"), "e_proj",
                   std::vector<std::string>{"refine_feat"},
                   std::vector<std::string>{"proj"}, autocastFp16);
    E.refine = std::make_unique<TorchEngine>(P("e_refine"), "e_refine",
                   std::vector<std::string>{"net", "inp", "corr", "flow"},
                   std::vector<std::string>{"net", "up_masks", "residual_flows"}, autocastFp16);
    return E;
}

// ---------------------------------------------------------------------------
// resnet norm constants (ImageNet), shape (1,1,3,1,1) to broadcast over B,T.
// ---------------------------------------------------------------------------
static torch::Tensor resnetMean(torch::Device dev) {
    return torch::tensor({0.485, 0.456, 0.406},
               torch::TensorOptions().dtype(torch::kFloat).device(dev)).view({1, 1, 3, 1, 1});
}
static torch::Tensor resnetStd(torch::Device dev) {
    return torch::tensor({0.229, 0.224, 0.225},
               torch::TensorOptions().dtype(torch::kFloat).device(dev)).view({1, 1, 3, 1, 1});
}

// ---------------------------------------------------------------------------
// InputPadderMF (mode "sintel", padding_factor 8) — basic.py.
// F.pad order is (W_left,W_right, H_top,H_bot, C0,C0) on the last-3 dims.
// ---------------------------------------------------------------------------
struct InputPadderMF {
    std::vector<int64_t> pad_;   // [wl, wr, ht, hb, 0, 0]
    InputPadderMF(int64_t H, int64_t W, int64_t pf = 8) {
        int64_t pad_ht = (((H / pf) + 1) * pf - H) % pf;
        int64_t pad_wd = (((W / pf) + 1) * pf - W) % pf;
        pad_ = { pad_wd / 2, pad_wd - pad_wd / 2,
                 pad_ht / 2, pad_ht - pad_ht / 2, 0, 0 };
    }
    torch::Tensor pad(const torch::Tensor& x) const {
        return Fn::pad(x, Fn::PadFuncOptions(pad_).mode(torch::kReplicate));
    }
    torch::Tensor unpad(const torch::Tensor& x) const {
        const int64_t h = x.size(-2), w = x.size(-1);
        const int64_t c0 = pad_[2], c1 = h - pad_[3], c2 = pad_[0], c3 = w - pad_[1];
        return x.index({Ellipsis, Slice(c0, c1), Slice(c2, c3)});
    }
};

// ---------------------------------------------------------------------------
// custom_interpolate (megaflow.py) — bilinear, INT_MAX-chunked. align_corners=true.
// ---------------------------------------------------------------------------
static torch::Tensor customInterpolate(const torch::Tensor& x,
                                       std::vector<int64_t> size, bool align_corners = true)
{
    const int64_t INT_MAX_ = 1610612736;
    const int64_t input_elements = size[0] * size[1] * x.size(0) * x.size(1);
    auto opt = Fn::InterpolateFuncOptions().size(size)
                   .mode(torch::kBilinear).align_corners(align_corners);
    if (input_elements > INT_MAX_) {
        const int64_t nchunks = input_elements / INT_MAX_ + 1;
        auto chunks = torch::chunk(x, nchunks, 0);
        std::vector<torch::Tensor> out;
        out.reserve(chunks.size());
        for (auto& ch : chunks) out.push_back(Fn::interpolate(ch, opt));
        return torch::cat(out, 0).contiguous();
    }
    return Fn::interpolate(x, opt);
}

// ---------------------------------------------------------------------------
// get_T_padded_images (non-training): compute window start indices and replicate
// the last frame to pad T up to indices.back()+seq_len.
// ---------------------------------------------------------------------------
static torch::Tensor getTPaddedImages(const torch::Tensor& images_in, int64_t origin_T,
                                      int64_t seq_len, int64_t stride,
                                      std::vector<int64_t>& indices_out, int64_t& T_out)
{
    auto images = images_in;
    const int64_t B = images.size(0), C = images.size(2),
                  H = images.size(3), W = images.size(4);
    std::vector<int64_t> indices;
    int64_t start = 0;
    while (start + seq_len < origin_T) { indices.push_back(start); start += stride; }
    indices.push_back(start);
    const int64_t Tpad = indices.back() + seq_len - origin_T;
    int64_t T = origin_T;
    if (Tpad > 0) {
        auto last = images.index({Slice(), Slice(origin_T - 1, origin_T)});   // [B,1,C,H,W]
        auto padt = last.expand({B, Tpad, C, H, W});
        images = torch::cat({images, padt}, 1);
        T = origin_T + Tpad;
    }
    indices_out = std::move(indices);
    T_out = T;
    return images.contiguous();
}

// ---------------------------------------------------------------------------
// forwardTrack — windowed solver. imgs_in: (B,origin_T,3,H,W) float, 0..255, RGB.
// Returns flow_final (B,origin_T,2,H,W) in NATIVE pixel units.
// Assumes an outer autocast scope is active (caller's responsibility).
// ---------------------------------------------------------------------------
torch::Tensor forwardTrack(Engines& E, const torch::Tensor& imgs_in,
                           int num_reg_refine, cudaStream_t stream)
{
    const int64_t S  = E.seq_len;          // 16
    const int64_t rf = E.refine_factor;    // 4
    const int64_t patch = E.patch_size;    // 14
    const auto dev = imgs_in.device();

    const int64_t B = imgs_in.size(0), origin_T = imgs_in.size(1), C = imgs_in.size(2);
    const int64_t H = imgs_in.size(3), W = imgs_in.size(4);

    // normalize: (imgs/255 - mean)/std
    auto imgs = ((imgs_in / 255.0 - resnetMean(dev)) / resnetStd(dev)).contiguous();

    const int64_t stride = S / 2;
    std::vector<int64_t> indices; int64_t T = 0;
    imgs = getTPaddedImages(imgs, origin_T, S, stride, indices, T);

    InputPadderMF padder(H, W);
    imgs = padder.pad(imgs);                                  // [B,T,C,H_pad,W_pad]
    const int64_t H_pad = imgs.size(3), W_pad = imgs.size(4);

    // fix_width: resize_w=518 (patch 14); resize_h = round(H_pad*(rw/W_pad)/patch)*patch
    const int64_t resize_w = (patch == 14) ? 518 : 592;
    const int64_t resize_h = (int64_t)std::llround(
        (double)H_pad * ((double)resize_w / (double)W_pad) / (double)patch) * patch;

    auto imgs_flat = imgs.view({B * T, C, H_pad, W_pad});
    auto resize_input = Fn::interpolate(imgs_flat,
            Fn::InterpolateFuncOptions().size(std::vector<int64_t>{resize_h, resize_w})
                .mode(torch::kBilinear).align_corners(true))
        .view({B, T, C, resize_h, resize_w});

    // --- the three "once" engines: e_vit, e_cnn, e_head ---
    auto tokens = E.vit->run({{"img", resize_input}}, stream).at("tokens");   // [24,1,T,P,Cembed]
    auto cnnout = E.cnn->run({{"frames", imgs_flat}}, stream);
    auto cnn0 = cnnout.at("cnn0");
    auto cnn1 = cnnout.at("cnn1");
    auto features = E.head->run({{"tokens", tokens}, {"img", resize_input},
                                 {"cnn0", cnn0}, {"cnn1", cnn1}}, stream).at("feat");  // [B,T,Cf,hf,wf]

    auto feature0 = features.index({Slice(), 0}).contiguous();                // [B,Cf,hf,wf]
    // fuse_cnn: use the LAST FPN level (cnn1), reshaped to [B,T,...]
    auto cnn_feat = cnn1.view({B, T, cnn1.size(1), cnn1.size(2), cnn1.size(3)}).contiguous();
    auto feature0_cnn = cnn_feat.index({Slice(), 0}).contiguous();            // [B,128,hc,wc]

    at::autocast::clear_cache();   // drop the ViT weight-cast cache (run once)

    const int64_t Cf = features.size(2), Hf = features.size(3), Wf = features.size(4);
    const int64_t Hr = H_pad / rf, Wr = W_pad / rf;

    auto full_flows = torch::zeros({B, T, 2, H, W},
                          torch::TensorOptions().dtype(imgs.dtype()).device(dev));

    for (size_t ii = 0; ii < indices.size(); ++ii) {
        const int64_t ind = indices[ii];
        auto feat_chunk     = features.index({Slice(), Slice(ind, ind + S)});  // [B,S,Cf,hf,wf]
        auto feat_chunk_cnn = cnn_feat.index({Slice(), Slice(ind, ind + S)});  // [B,S,128,hc,wc]

        // flows_init: per-frame global correlation vs the fixed anchor feature0
        std::vector<torch::Tensor> flows_init;
        flows_init.reserve(S);
        for (int64_t t = 0; t < S; ++t)
            flows_init.push_back(globalCorrelationSoftmax(feature0,
                                     feat_chunk.index({Slice(), t})));
        auto flows0 = torch::stack(flows_init, 1).contiguous();                // [B,S,2,hf,wf]

        // to 1/rf grid
        auto flows = resizeFlowBilinear(flows0, Hr, Wr, flows0.size(-2), flows0.size(-1)).contiguous();

        auto refine_feature = customInterpolate(
            feat_chunk.reshape({B * S, Cf, Hf, Wf}), std::vector<int64_t>{Hr, Wr}, true);
        auto proj = E.proj->run({{"refine_feat", refine_feature}}, stream).at("proj");
        auto pchunks = torch::chunk(proj, 2, 1);
        auto net = pchunks[0].contiguous();
        auto inp = pchunks[1].contiguous();

        torch::Tensor flow_up_final;
        for (int it = 0; it < num_reg_refine; ++it) {
            flows = flows.detach();
            std::vector<torch::Tensor> corrs;
            corrs.reserve(S);
            torch::Tensor correlation;
            for (int64_t t = 0; t < S; ++t) {
                correlation = localCorrelationWithFlow(
                    feature0_cnn, feat_chunk_cnn.index({Slice(), t}),
                    flows.index({Slice(), t}), 4);
                corrs.push_back(correlation);
            }
            auto corrs_t = torch::stack(corrs, 1)
                               .view({-1, correlation.size(1), correlation.size(2), correlation.size(3)});
            auto flows_flat = flows.view({-1, flows.size(2), flows.size(3), flows.size(4)});

            auto rout = E.refine->run({{"net", net}, {"inp", inp},
                                       {"corr", corrs_t}, {"flow", flows_flat}}, stream);
            net = rout.at("net");
            auto up_masks  = rout.at("up_masks");
            auto residual  = rout.at("residual_flows");
            flows = flows + residual.view(flows.sizes());

            if (it == num_reg_refine - 1) {
                auto flow_up = upsampleFlowWithMask(
                    flows.view({-1, flows.size(2), flows.size(3), flows.size(4)}),
                    up_masks, (int)rf);                                   // [B*S,2,H_pad,W_pad]
                auto flow_up5 = flow_up.view({B, S, 2, flow_up.size(2), flow_up.size(3)});
                flow_up_final = padder.unpad(flow_up5);                   // [B,S,2,H,W]
            }
        }
        full_flows.index_put_({Slice(), Slice(ind, ind + S)},
                              flow_up_final.reshape({B, S, 2, H, W}));
    }

    return full_flows.index({Slice(), Slice(None, origin_T)}).contiguous();
}

// ---------------------------------------------------------------------------
// compute dtype: bf16 on Ampere+ (cc>=8), else fp16 — matches _run_megaflow's
// `bf16 if torch.cuda.is_bf16_supported() else fp16`.
// ---------------------------------------------------------------------------
static at::ScalarType computeDtype()
{
    int dev = 0; cudaGetDevice(&dev);
    int major = 0; cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return (major >= 8) ? at::kBFloat16 : at::kHalf;
}

// identity trajectory = absolute coord grid (1,2,H,W). Mirrors _identity_traj
// (gridcloud2d -> permute -> reshape). NOTE: assumes gridcloud2d's (x,y) ordering
// matches coords_grid; pending gridcloud2d source to confirm.
static torch::Tensor identityTraj(int64_t H, int64_t W, torch::Device dev)
{
    return coordsGrid(1, H, W, dev).to(torch::kFloat);   // (1,2,H,W)
}

// ---------------------------------------------------------------------------
// _run_megaflow: (1,len,3,H,W) 0..255 -> (len,2,H,W) absolute coords. bf16
// autocast around forwardTrack (matches _run_megaflow), then + coord grid.
// ---------------------------------------------------------------------------
static torch::Tensor runMegaflow(Engines& E, const torch::Tensor& frames_5d,
                                 int iters, cudaStream_t stream)
{
    const int64_t H = frames_5d.size(3), W = frames_5d.size(4);
    const auto dev = frames_5d.device();
    const at::ScalarType dt = computeDtype();

    at::autocast::set_autocast_dtype(at::kCUDA, dt);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    auto flow_final = forwardTrack(E, frames_5d, iters, stream);     // (1,len,2,H,W)
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    at::autocast::clear_cache();

    auto grid = coordsGrid(1, H, W, dev).view({1, 1, 2, H, W}).to(torch::kFloat);
    auto traj = flow_final.to(torch::kFloat) + grid;                 // (1,len,2,H,W)
    return traj.index({0}).contiguous();                             // (len,2,H,W)
}

// ---------------------------------------------------------------------------
// build_full_trajectory: forward [ref..T) + backward [ref..0], flip backward,
// concat, force identity at ref.
// ---------------------------------------------------------------------------
SolveResult solveTrajectory(Engines& E, const torch::Tensor& frames_model,
                            const SolveConfig& cfg, cudaStream_t stream,
                            ProgressFn progress)
{
    torch::InferenceMode im;
    const int64_t T  = frames_model.size(0);
    const int64_t mH = frames_model.size(2), mW = frames_model.size(3);
    // frames_model may live on CPU (the caller can offload the full stack to host
    // RAM — it grows with frame count). Compute happens on CUDA regardless; we
    // move only each pass's gathered frames to the GPU just before runMegaflow,
    // so the whole input stack never has to be resident in VRAM at once.
    const auto dev = torch::Device(torch::kCUDA);
    const auto idx_dev = frames_model.device();   // gather index on the stack's device
    const int64_t ref = cfg.ref_idx;
    auto lopt = torch::TensorOptions().dtype(torch::kLong).device(idx_dev);

    // Optional CUDA memory report (set MEGAFLOW_MEMREPORT=1). Prints VRAM in-use
    // per phase so we can see WHICH phase dominates (forward vs backward, whether
    // the reclaim helps) before committing to bigger streaming work. Uses
    // cudaMemGetInfo only (no version-fragile allocator-stats struct); the number
    // matches nvidia-smi, so it's directly comparable to the logs.
    const bool memreport = std::getenv("MEGAFLOW_MEMREPORT") != nullptr;
    auto report = [&](const char* tag) {
        if (!memreport) return;
        cudaDeviceSynchronize();
        size_t freeB = 0, totalB = 0; cudaMemGetInfo(&freeB, &totalB);
        fprintf(stderr, "[MEMREPORT] %-18s VRAM used = %.2f GB / %.2f GB\n", tag,
                (double)(totalB - freeB) / 1e9, (double)totalB / 1e9);
        fflush(stderr);
    };
    report("start");

    // Forward: frames [ref, ref+1, ..., T-1]
    torch::Tensor fwd;
    if (T - ref >= 2) {
        auto idx = torch::arange(ref, T, lopt);
        auto pass = frames_model.index_select(0, idx).unsqueeze(0).to(dev);   // -> GPU
        fwd = runMegaflow(E, pass, cfg.iters, stream);
        if (progress && !progress(0.5f)) return {};
    } else {
        fwd = identityTraj(mH, mW, dev);                  // (1,2,H,W)
    }
    report("after forward");

    // Reclaim the forward pass's working set (features/full_flows for that pass)
    // BEFORE the backward pass allocates its own — they don't need to coexist, so
    // this directly lowers the PEAK, not just the resting footprint.
    c10::cuda::CUDACachingAllocator::emptyCache();
    report("after fwd reclaim");

    // Backward: frames [ref, ref-1, ..., 0]
    torch::Tensor bwd;
    if (ref + 1 >= 2) {
        auto idx = torch::arange(ref, -1, -1, lopt);
        auto pass = frames_model.index_select(0, idx).unsqueeze(0).to(dev);   // -> GPU
        bwd = runMegaflow(E, pass, cfg.iters, stream);
        if (progress && !progress(1.0f)) return {};
    } else {
        bwd = identityTraj(mH, mW, dev);                  // (1,2,H,W)
    }
    report("after backward");

    // full = cat([flip(bwd,0), fwd[1:]], 0); full[ref] = identity
    auto full = torch::cat({torch::flip(bwd, {0}),
                            fwd.index({Slice(1, None)})}, 0);         // (T,2,H,W)
    full.index_put_({ref}, identityTraj(mH, mW, dev).index({0}));

    SolveResult r;
    r.traj_maps = full.to(torch::kCPU).contiguous();
    r.model_h = (int)mH;
    r.model_w = (int)mW;
    return r;
}

} // namespace megaflow
