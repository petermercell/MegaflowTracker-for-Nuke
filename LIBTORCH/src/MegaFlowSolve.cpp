// ============================================================================
// MegaFlowSolve.cpp  —  Nuke Iop wrapping the validated megaflow::solveTrajectory.
// See MegaFlowSolve.h for the design. Mirrors CutieRoto's button/pull/progress/
// stream conventions; the solve math is the already-proven core (0.0000px vs the
// Python oracle through the mf_solve CLI).
// ============================================================================
#include "MegaFlowSolve.h"
#include "cuda_utils.h"   // cudautils::emptyCache / reset / memInfoString (shared VRAM helper)

#include <torch/torch.h>
#include <opencv2/imgproc.hpp>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>

// ----------------------------------------------------------------------------
MegaFlowSolve::MegaFlowSolve(Node* node)
    : Iop(node),
      enginesDirKnob_(""),
      npzPathKnob_(""),
      refFrame_(1),
      firstFrame_(0),
      lastFrame_(0),
      iters_(8),
      fixWidth_(518),
      colorspace_(0),
      gpuDevice_(0)
{}

MegaFlowSolve::~MegaFlowSolve() {}

const char* MegaFlowSolve::node_help() const {
    return "MegaFlowSolve — native MegaFlow trajectory solver.\n\n"
           "Connect a plate to input 0, set the Reference Frame and an output "
           ".npz path, then press Solve. Writes a trajectory cache that the "
           "MEGAFlowTracker_PM gizmo reads via its 'NPZ FILE' knob.\n\n"
           "The image output is a straight passthrough of input 0.\n\n"
           "MegaFlowSolve for Nuke by Peter Mercell, 2026 — petermercell.com\n"
           "MegaFlow by the cvg authors (Apache-2.0).";
}

// ----------------------------------------------------------------------------
bool MegaFlowSolve::test_input(int n, Op* op) const {
    return dynamic_cast<Iop*>(op) != nullptr;
}

const char* MegaFlowSolve::input_label(int n, char* buf) const {
    return "";   // plate (unlabeled, B-side)
}

// ----------------------------------------------------------------------------
// knobs
// ----------------------------------------------------------------------------
void MegaFlowSolve::knobs(Knob_Callback f) {
    File_knob(f, &enginesDirKnob_, "engines_dir", "Engines Dir");
    Tooltip(f, "Folder containing the 5 traced .pt engines (e_vit, e_cnn, e_head,\n"
               "e_proj, e_refine). Empty = the MEGAFLOW_ENGINES environment variable.");

    File_knob(f, &npzPathKnob_, "npz", "Output NPZ");
    Tooltip(f, "Where to write the trajectory cache. Point the MEGAFlowTracker_PM\n"
               "gizmo's 'NPZ FILE' knob at this same path.");

    Int_knob(f, &refFrame_, "ref_frame", "Reference Frame");
    Tooltip(f, "The frame the trajectories are anchored to (NUKE frame number).\n"
               "Pick a frame where your tracked features are clearly visible.");

    Int_knob(f, &firstFrame_, "first_frame", "First Frame");
    Tooltip(f, "First frame to solve. 0 = the input's first frame.");
    ClearFlags(f, Knob::STARTLINE);
    Int_knob(f, &lastFrame_, "last_frame", "Last Frame");
    Tooltip(f, "Last frame to solve. 0 = the input's last frame.");
    ClearFlags(f, Knob::STARTLINE);

    Button(f, "solve", "Solve");
    Tooltip(f, "Pull the plate over the frame range, run the MegaFlow forward+\n"
               "backward solve, and write the .npz cache.");

    Divider(f, "");
    static const char* const cs_modes[] = { "linear", "sRGB / raw", nullptr };
    Enumeration_knob(f, &colorspace_, cs_modes, "colorspace", "Input Colorspace");
    Tooltip(f, "How input 0's RGB is encoded. 'linear' (default) applies the same\n"
               "linear->sRGB encode the reference pipeline used for EXRs. Choose\n"
               "'sRGB / raw' if the input is already display-encoded (e.g. a Read\n"
               "set to sRGB, or 8-bit footage) so it is passed through unchanged.");

    Int_knob(f, &iters_, "iters", "Refine Iterations");
    Tooltip(f, "num_reg_refine — refine iterations per window. 8 matches the\n"
               "validated reference. Fewer = faster but looser far from ref.");
    SetRange(f, 1, 12);

    Int_knob(f, &fixWidth_, "fix_width", "Model Width");
    Tooltip(f, "Model resize target width (calculate_dynamic_size). 518 = the\n"
               "validated default; change only if you know the model variant.");
    SetFlags(f, Knob::HIDDEN);

    Int_knob(f, &gpuDevice_, "gpu", "GPU Device");
    SetFlags(f, Knob::HIDDEN);

    Divider(f, "");
    Button(f, "free_vram", "Free VRAM");
    Tooltip(f, "Release libtorch's cached-but-unused VRAM now, without re-solving.\n"
               "Engines stay resident, so the next Solve is still fast.");
    Button(f, "release_engines", "Release Engines");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Drop the loaded engines entirely and hard-reclaim VRAM (the biggest\n"
               "free). The next Solve reloads the 5 .pt — e_vit alone is 4.6GB.");

    Divider(f, "");
    Text_knob(f, "MegaFlowSolve for Nuke by Peter Mercell, 2026\n"
                 "petermercell.com\n\n"
                 "MegaFlow by the cvg authors (Apache-2.0)");
}

int MegaFlowSolve::knob_changed(Knob* k) {
    if (!k) return Iop::knob_changed(k);
    const std::string name = k->name();
    if (name == "solve") {
        doSolve();
        return 1;
    }
    if (name == "free_vram") {
        cudautils::emptyCache();
        progressMessage("MegaFlowSolve: freed cache — %s",
                        cudautils::memInfoString().c_str());
        return 1;
    }
    if (name == "release_engines") {
        // Drop the engines AND clear the readiness flags together. Resetting
        // engines_ alone would leave enginesReady_/loadedFrom_ stale, so the next
        // Solve's ensureEngines() early-outs on (enginesReady_ && loadedFrom_==dir)
        // and then dereferences a now-null engines_. Clearing both forces a clean
        // reload.
        engines_.reset();
        enginesReady_ = false;
        loadedFrom_.clear();
        cudautils::reset();               // emptyCache + device synchronize
        progressMessage("MegaFlowSolve: engines released — %s",
                        cudautils::memInfoString().c_str());
        return 1;
    }
    return Iop::knob_changed(k);
}

// ----------------------------------------------------------------------------
// passthrough: input 0 straight through. No added channels, no cache.
// ----------------------------------------------------------------------------
void MegaFlowSolve::_validate(bool for_real) {
    if (input(0)) {
        input(0)->validate(for_real);
        info_ = input(0)->info();
        fX_ = info_.x(); fY_ = info_.y(); fW_ = info_.w(); fH_ = info_.h();
    } else {
        info_.set(0, 0, 512, 512);
    }
}

void MegaFlowSolve::_request(int x, int y, int r, int t, ChannelMask m, int count) {
    if (input(0)) input(0)->request(x, y, r, t, m, count);
}

void MegaFlowSolve::engine(int y, int x, int r, ChannelMask m, Row& row) {
    if (input(0)) {
        input(0)->get(y, x, r, m, row);
    } else {
        foreach(z, m) {
            float* o = row.writable(z) + x;
            std::memset(o, 0, sizeof(float) * (r - x));
        }
    }
}

// ----------------------------------------------------------------------------
// engines: resolve dir, build the libtorch stream from libtorch's pool (one
// CUDA runtime — the Windows-safe discipline from CutieRoto::buildPipeline),
// loadEngines once. Reload if the dir knob changed.
// ----------------------------------------------------------------------------
std::string MegaFlowSolve::resolveEnginesDir() const {
    if (enginesDirKnob_ && enginesDirKnob_[0]) return enginesDirKnob_;
    const char* env = std::getenv("MEGAFLOW_ENGINES");
    if (env && env[0]) return env;
    return "";
}

void MegaFlowSolve::ensureEngines() {
    std::string dir = resolveEnginesDir();
    if (dir.empty())
        throw std::runtime_error(
            "no engines directory: set the 'Engines Dir' knob or the "
            "MEGAFLOW_ENGINES environment variable");

    if (enginesReady_ && loadedFrom_ == dir) return;

    if (!torch::cuda::is_available())
        throw std::runtime_error("CUDA is not available (no usable GPU / driver)");
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0)
        throw std::runtime_error("no CUDA devices reported");
    if (gpuDevice_ < 0 || gpuDevice_ >= deviceCount)
        throw std::runtime_error("invalid GPU device " + std::to_string(gpuDevice_));
    cudaSetDevice(gpuDevice_);

    // Stream from libtorch's own pool so it lives in libtorch's CUDA runtime
    // (handing a toolkit-created stream to libtorch crashes across CRTs on
    // Windows). Same as CutieRoto's libtorch backend.
    auto torchStream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, gpuDevice_);
    c10::cuda::setCurrentCUDAStream(torchStream);
    stream_ = torchStream.stream();

    // autocastFp16=false: the solve core sets its own outer bf16 autocast scope
    // around forwardTrack and the engines inherit it (validated in mf_solve).
    engines_ = std::make_unique<megaflow::Engines>(
        megaflow::loadEngines(dir, stream_, /*autocastFp16=*/false));
    loadedFrom_ = dir;
    enginesReady_ = true;
}

// ----------------------------------------------------------------------------
// default solve range from input 0's frame range. The explicit First/Last Frame
// knobs override this in doSolve when set (>0). first_frame()/last_frame() live
// on Iop (via IopInfoOwner), not Op — so cast the input first.
// ----------------------------------------------------------------------------
bool MegaFlowSolve::inputFrameRange(int& first, int& last) const {
    Iop* iop = dynamic_cast<Iop*>(input(0));
    if (iop) {
        iop->validate(false);
        int f0 = iop->first_frame();
        int f1 = iop->last_frame();
        if (f1 < f0) std::swap(f0, f1);
        if (f1 > f0) { first = f0; last = f1; return true; }
    }
    int cur = (int)outputContext().frame();
    first = cur; last = cur;
    return false;
}

// ----------------------------------------------------------------------------
// pull input 0 at `frame` -> (3,H,W) CPU float, top-down, values 0..255.
//
// Mirrors CutieRoto::pullInputResized's row handling (Nuke rows are bottom-up;
// flip to top-down so the orientation matches OpenCV imread, which is what the
// CLI/oracle used), then applies the _float_rgb_to_uint8 encode:
//   linear mode : v = clamp(lin->sRGB(max(v,0)), 0,1); out = floor(v*255 + 0.5)
//   sRGB/raw    : v = clamp(max(v,0), 0,1);            out = floor(v*255 + 0.5)
// Returns the *rounded* 0..255 values as float so the downstream resize sees the
// same uint8-quantised pixels cv2.resize saw in the CLI.
// ----------------------------------------------------------------------------
static inline float lin2srgb(float c) {
    if (c < 0.f) c = 0.f;
    return (c <= 0.0031308f) ? c * 12.92f
                             : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// Pull input 0 at `frame` -> native-res RGB uint8 cv::Mat (HWC, top-down),
// IDENTICAL to the CLI's get_native_frames + _float_rgb_to_uint8 output, so the
// subsequent cv::resize(INTER_LINEAR) in doSolve reproduces the CLI byte-for-byte.
// Nuke rows are bottom-up -> flip to top-down (matches OpenCV imread orientation).
cv::Mat MegaFlowSolve::pullNativeRGB255(double frame, int& nativeH, int& nativeW) {
    OutputContext oc = outputContext();
    oc.setFrame(frame);
    Op* op = inputAt(0, oc);
    if (!op) return cv::Mat();
    Iop* iop = dynamic_cast<Iop*>(op);
    if (!iop) return cv::Mat();

    iop->validate(true);
    Box b = iop->info().box();
    const int w = b.w(), h = b.h(), x0 = b.x(), y0 = b.y();
    if (w <= 0 || h <= 0) return cv::Mat();

    iop->request(x0, y0, x0 + w, y0 + h, Mask_RGB, 1);

    const bool linear = (colorspace_ == 0);
    cv::Mat out(h, w, CV_8UC3);                       // HWC, RGB, uint8 — like the CLI
    Channel rgb[3] = { Chan_Red, Chan_Green, Chan_Blue };
    for (int yy = 0; yy < h; ++yy) {
        Row row(x0, x0 + w);
        iop->get(y0 + yy, x0, x0 + w, Mask_RGB, row);
        const int dy = h - 1 - yy;                    // flip bottom-up -> top-down
        unsigned char* dp = out.ptr<unsigned char>(dy);
        const float* sr = row[rgb[0]] + x0;
        const float* sg = row[rgb[1]] + x0;
        const float* sb = row[rgb[2]] + x0;
        for (int xi = 0; xi < w; ++xi) {
            float vr = sr[xi], vg = sg[xi], vb = sb[xi];
            if (linear) { vr = lin2srgb(vr); vg = lin2srgb(vg); vb = lin2srgb(vb); }
            else { if (vr < 0.f) vr = 0.f; if (vg < 0.f) vg = 0.f; if (vb < 0.f) vb = 0.f; }
            if (vr > 1.f) vr = 1.f; if (vg > 1.f) vg = 1.f; if (vb > 1.f) vb = 1.f;
            dp[xi * 3 + 0] = (unsigned char)(vr * 255.0f + 0.5f);   // R
            dp[xi * 3 + 1] = (unsigned char)(vg * 255.0f + 0.5f);   // G
            dp[xi * 3 + 2] = (unsigned char)(vb * 255.0f + 0.5f);   // B
        }
    }

    nativeH = h; nativeW = w;
    return out;
}

// ----------------------------------------------------------------------------
// the Solve button action: pull the range, build (T,3,mH,mW) 0..255 like the
// CLI, run solveTrajectory under a progress task, write the gizmo-compatible npz.
// ----------------------------------------------------------------------------
void MegaFlowSolve::doSolve() {
    if (!input(0)) { Op::error("MegaFlowSolve: no plate on input 0"); return; }

    std::string npzPath = (npzPathKnob_ && npzPathKnob_[0]) ? npzPathKnob_ : "";
    if (npzPath.empty()) { Op::error("MegaFlowSolve: set an Output NPZ path"); return; }

    try { ensureEngines(); }
    catch (const std::exception& e) { Op::error("MegaFlowSolve engines: %s", e.what()); return; }

    // frame range
    int rangeFirst = 0, rangeLast = 0;
    inputFrameRange(rangeFirst, rangeLast);
    int first = (firstFrame_ > 0) ? firstFrame_ : rangeFirst;
    int last  = (lastFrame_  > 0) ? lastFrame_  : rangeLast;
    if (last < first) std::swap(first, last);
    const int T = last - first + 1;
    if (T < 2) { Op::error("MegaFlowSolve: need >= 2 frames (got %d)", T); return; }

    const int nukeFirst = first;
    const int refFrame  = refFrame_;
    const int refIdx    = refFrame - nukeFirst;
    if (refIdx < 0 || refIdx >= T) {
        Op::error("MegaFlowSolve: Reference Frame %d is outside the solve range %d..%d",
                  refFrame, first, last);
        return;
    }

    progressMessage("MegaFlowSolve: reading %d frames...", T);

    // ---- pull every frame -> native RGB 0..255 -> model-res (T,3,mH,mW) ----
    int nativeH = 0, nativeW = 0, mh = 0, mw = 0;
    std::vector<torch::Tensor> fr;
    fr.reserve(T);
    for (int i = 0; i < T; ++i) {
        if (aborted()) { progressDismiss(); return; }
        int nh = 0, nw = 0;
        cv::Mat rgb;
        try { rgb = pullNativeRGB255((double)(first + i), nh, nw); }
        catch (const std::exception& e) {
            progressDismiss();
            Op::error("MegaFlowSolve: pull frame %d: %s", first + i, e.what()); return;
        }
        if (rgb.empty()) { progressDismiss(); Op::error("MegaFlowSolve: failed to read frame %d", first + i); return; }

        // DIAGNOSTIC: dump the native-res uint8 RGB of the ref frame so it can be
        // compared pixel-for-pixel against what the CLI (cv2.imread + encode) built.
        // Gated on MEGAFLOW_DUMP (set to a path prefix). Format: "MFD1" magic,
        // int32 w, int32 h, then h*w*3 uint8 (RGB, top-down).
        if (const char* dp = std::getenv("MEGAFLOW_DUMP")) {
            if (i == refIdx) {
                std::string path = std::string(dp) + "_nuke_native_ref.bin";
                FILE* fp = std::fopen(path.c_str(), "wb");
                if (fp) {
                    int32_t hdr[3] = { 0x3144464D /*"MFD1"*/, nw, nh };
                    std::fwrite(hdr, sizeof(int32_t), 3, fp);
                    std::fwrite(rgb.data, 1, (size_t)nw * nh * 3, fp);  // HWC RGB uint8
                    std::fclose(fp);
                    progressMessage("MegaFlowSolve: dumped %s", path.c_str());
                }
            }
        }

        if (i == 0) {
            nativeH = nh; nativeW = nw;
            // calculate_dynamic_size: mw = fix_width; mh = round(nh*(mw/nw)/14)*14
            mw = fixWidth_;
            mh = (int)std::llround((double)nativeH * ((double)mw / (double)nativeW) / 14.0) * 14;
        } else if (nh != nativeH || nw != nativeW) {
            progressDismiss();
            Op::error("MegaFlowSolve: frame %d size %dx%d != first %dx%d "
                      "(all frames must share one format)", first + i, nw, nh, nativeW, nativeH);
            return;
        }

        // resize uint8 -> model res with cv::INTER_LINEAR — byte-IDENTICAL to the
        // CLI's cv2.resize (same algorithm, same uint8 re-quantization). This is
        // why the node links OpenCV imgproc: to reuse the exact resize the oracle
        // used, removing resize as a source of divergence.
        cv::Mat r;
        cv::resize(rgb, r, cv::Size(mw, mh), 0, 0, cv::INTER_LINEAR);
        // HWC uint8 -> CHW float 0..255 (matches mf_solve's from_blob.to(float).permute)
        // Keep on CPU: solveTrajectory streams each pass to the GPU, so the full
        // input stack never has to be VRAM-resident (matters for long shots).
        auto t = torch::from_blob(r.data, {mh, mw, 3}, torch::kUInt8)
                     .to(torch::kFloat).permute({2, 0, 1}).contiguous();
        fr.push_back(t.clone());                             // (3,mh,mw) on CPU (clone: r.data is reused)

        progressFraction((double)(i + 1) / (double)T * 0.15, Op::StatusModal);  // pulls = first 15%
        progressMessage("MegaFlowSolve: read frame %d/%d", i + 1, T);
    }

    auto frames_model = torch::stack(fr, 0).contiguous();    // (T,3,mh,mw) on CPU host RAM

    // ---- solve (forward + backward) ----
    megaflow::SolveConfig cfg;
    cfg.ref_idx   = refIdx;
    cfg.iters     = iters_;
    cfg.fix_width = fixWidth_;

    megaflow::SolveResult res;
    try {
        res = megaflow::solveTrajectory(
            *engines_, frames_model, cfg, stream_,
            [&](float p) -> bool {
                // map solve [0,1] to the 15..98% band; honour cancel
                progressFraction(0.15 + 0.83 * (double)p, Op::StatusModal);
                return !aborted();
            });
    } catch (const std::exception& e) {
        progressDismiss();
        Op::error("MegaFlowSolve solve: %s", e.what()); return;
    }

    // ---- meta JSON — keys the gizmo reads (load_cache/sample_to_nk) ----
    std::ostringstream m;
    m << "{"
      << "\"version\": 1, "
      << "\"nuke_first_frame\": " << nukeFirst << ", "
      << "\"last_frame\": " << last << ", "
      << "\"ref_frame\": " << refFrame << ", "
      << "\"ref_idx\": " << refIdx << ", "
      << "\"native_h\": " << nativeH << ", "
      << "\"native_w\": " << nativeW << ", "
      << "\"model_h\": " << res.model_h << ", "
      << "\"model_w\": " << res.model_w << ", "
      << "\"fix_width\": " << fixWidth_ << ", "
      << "\"iters\": " << iters_ << ", "
      << "\"dtype\": \"float16\", "
      << "\"n_frames\": " << T
      << "}";

    try {
        megaflow::saveCache(npzPath, res.traj_maps, m.str());
    } catch (const std::exception& e) {
        progressDismiss();
        Op::error("MegaFlowSolve save: %s", e.what()); return;
    }

    progressMessage("MegaFlowSolve: wrote %s", npzPath.c_str());

    // Self-clean: return this Solve's cached-but-unused VRAM to the driver. This
    // complements the in-solve reclaim (solveTrajectory empties once BETWEEN the
    // fwd/bwd passes, so it never sees the backward working set) by clearing
    // what's left after the whole solve. Engines stay resident — they're live
    // allocations, not cached free blocks — so the next Solve doesn't reload.
    cudautils::emptyCache();

    progressDismiss();
}

// ----------------------------------------------------------------------------
static Iop* build(Node* node) { return new MegaFlowSolve(node); }
const Iop::Description MegaFlowSolve::description("MegaFlowSolve", "AI/MegaFlowSolve", build);
