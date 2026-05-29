// ============================================================================
// megaflow_glue.cpp  —  pure-torch glue, transcribed op-for-op from the model.
//
// Sources (all verified against inspect.getsource):
//   coords_grid                 <- geometry.py
//   global_correlation_softmax  <- matching.py
//   upsample_flow_with_mask     <- model_utils.py
//   generate_window_grid        <- matching.py  (helper, static)
//   normalize_coords            <- matching.py  (helper, static)
//   local_correlation_with_flow <- matching.py
//   resize_flow_bilinear        <- megaflow.py
//
// Each public fn mirrors its Python op-for-op so it can be unit-diffed against
// the Python original on captured tensors before forwardTrack builds on it.
// ============================================================================
#include "megaflow_solve.h"

#include <torch/nn/functional.h>
#include <cmath>

namespace Fn = torch::nn::functional;
using torch::indexing::Slice;

namespace megaflow {

// --- geometry.py: coords_grid -------------------------------------------------
// y,x = meshgrid(arange(h), arange(w)) ['ij'];  stack([x,y]).float()[None].repeat(b)
torch::Tensor coordsGrid(int64_t b, int64_t h, int64_t w, torch::Device dev)
{
    auto lopts = torch::TensorOptions().dtype(torch::kLong);
    auto g = torch::meshgrid({torch::arange(h, lopts), torch::arange(w, lopts)}, "ij");
    auto y = g[0];                                           // [H,W] row idx
    auto x = g[1];                                           // [H,W] col idx
    auto grid = torch::stack({x, y}, 0).to(torch::kFloat);   // [2,H,W] (x,y)
    grid = grid.unsqueeze(0).repeat({b, 1, 1, 1});           // [B,2,H,W]
    return grid.to(dev);
}

// --- matching.py: global_correlation_softmax (pred_bidir_flow=False) ----------
// returns only `flow` (prob unused: use_self_attn_propagation=False)
torch::Tensor globalCorrelationSoftmax(const torch::Tensor& feature0,
                                       const torch::Tensor& feature1)
{
    const int64_t b = feature0.size(0), c = feature0.size(1),
                  h = feature0.size(2), w = feature0.size(3);
    auto f0 = feature0.view({b, c, -1}).permute({0, 2, 1});   // [B,HW,C]
    auto f1 = feature1.view({b, c, -1});                       // [B,C,HW]
    auto correlation = torch::matmul(f0, f1) / std::sqrt((double)c);   // [B,HW,HW]

    auto init_grid = coordsGrid(b, h, w, correlation.device());        // [B,2,H,W]
    auto grid = init_grid.view({b, 2, -1}).permute({0, 2, 1});         // [B,HW,2]

    auto prob = torch::softmax(correlation, -1);                       // [B,HW,HW]
    auto correspondence = torch::matmul(prob, grid)
                              .view({b, h, w, 2}).permute({0, 3, 1, 2});   // [B,2,H,W]
    return correspondence - init_grid;                                 // flow
}

// --- model_utils.py: upsample_flow_with_mask (is_depth=False) -----------------
torch::Tensor upsampleFlowWithMask(const torch::Tensor& flow,
                                   const torch::Tensor& up_mask, int factor)
{
    const int64_t b = flow.size(0), fc = flow.size(1),
                  h = flow.size(2), w = flow.size(3);
    auto mask = up_mask.view({b, 1, 9, factor, factor, h, w});
    mask = torch::softmax(mask, 2);

    auto up_flow = Fn::unfold(factor * flow,
                   Fn::UnfoldFuncOptions({3, 3}).padding(1));   // [B, fc*9, h*w]
    up_flow = up_flow.view({b, fc, 9, 1, 1, h, w});
    up_flow = (mask * up_flow).sum(2);                          // [B,fc,factor,factor,h,w]
    up_flow = up_flow.permute({0, 1, 4, 2, 5, 3});              // [B,fc,h,factor,w,factor]
    up_flow = up_flow.reshape({b, fc, factor * h, factor * w});
    return up_flow;
}

// --- matching.py: generate_window_grid (helper) -------------------------------
// x,y = meshgrid([linspace(w_min,w_max,len_w), linspace(h_min,h_max,len_h)]) ['ij']
// grid = stack((x,y),-1).transpose(0,1)  -> [len_h, len_w, 2]
static torch::Tensor generateWindowGrid(double h_min, double h_max,
                                        double w_min, double w_max,
                                        int64_t len_h, int64_t len_w, torch::Device dev)
{
    auto fopts = torch::TensorOptions().dtype(torch::kFloat).device(dev);
    auto lw = torch::linspace(w_min, w_max, len_w, fopts);
    auto lh = torch::linspace(h_min, h_max, len_h, fopts);
    auto g = torch::meshgrid({lw, lh}, "ij");        // each [len_w, len_h]
    auto x = g[0], y = g[1];
    return torch::stack({x, y}, -1).transpose(0, 1).to(torch::kFloat);   // [len_h,len_w,2]
}

// --- matching.py: normalize_coords (helper) -----------------------------------
// c = [(w-1)/2, (h-1)/2];  (coords - c) / c
static torch::Tensor normalizeCoords(const torch::Tensor& coords, int64_t h, int64_t w)
{
    auto c = torch::tensor({(double)(w - 1) / 2.0, (double)(h - 1) / 2.0},
                 torch::TensorOptions().dtype(torch::kFloat).device(coords.device()));
    return (coords - c) / c;
}

// --- matching.py: local_correlation_with_flow (padding_mode='zeros', dilation=1)
torch::Tensor localCorrelationWithFlow(const torch::Tensor& feature0,
                                       const torch::Tensor& feature1,
                                       const torch::Tensor& flow,
                                       int local_radius)
{
    const int64_t b = feature0.size(0), c = feature0.size(1),
                  h = feature0.size(2), w = feature0.size(3);
    auto coords_init = coordsGrid(b, h, w, feature0.device());     // [B,2,H,W]
    auto coords = coords_init.view({b, 2, -1}).permute({0, 2, 1});  // [B,HW,2]

    const int64_t local_h = 2 * local_radius + 1;
    const int64_t local_w = 2 * local_radius + 1;
    auto window_grid = generateWindowGrid(-local_radius, local_radius,
                                          -local_radius, local_radius,
                                          local_h, local_w, feature0.device());  // [2R+1,2R+1,2]
    // reshape(-1,2).repeat(b,1,1,1) -> [B,1,(2R+1)^2,2]  (explicit to avoid repeat-prepend ambiguity)
    window_grid = window_grid.reshape({-1, 2}).view({1, 1, local_h * local_w, 2})
                             .repeat({b, 1, 1, 1});
    auto sample_coords = coords.unsqueeze(-2) + window_grid;        // dilation=1; [B,HW,(2R+1)^2,2]
    sample_coords = sample_coords +
        flow.view({b, 2, -1}).permute({0, 2, 1}).unsqueeze(-2);     // + flow

    auto sample_coords_norm = normalizeCoords(sample_coords, h, w)
                                  .to(feature1.scalar_type()).contiguous();
    auto window_feature = Fn::grid_sample(feature1, sample_coords_norm,
                              Fn::GridSampleFuncOptions()
                                  .mode(torch::kBilinear)
                                  .padding_mode(torch::kZeros)
                                  .align_corners(true))
                              .permute({0, 2, 1, 3});               // [B,HW,C,(2R+1)^2]
    auto feature0_view = feature0.permute({0, 2, 3, 1}).view({b, h * w, 1, c});  // [B,HW,1,C]
    auto corr = torch::matmul(feature0_view, window_feature)
                    .view({b, h * w, -1}) / std::sqrt((double)c);   // [B,HW,(2R+1)^2]
    return corr.view({b, h, w, -1}).permute({0, 3, 1, 2}).contiguous();  // [B,(2R+1)^2,H,W]
}

// --- megaflow.py: resize_flow_bilinear ----------------------------------------
// interpolate to (ori_h,ori_w); channel0 *= ori_w/new_w, channel1 *= ori_h/new_h.
// handles 4D (B,2,H,W) and 5D (B,T,2,H,W).
torch::Tensor resizeFlowBilinear(const torch::Tensor& flow_in,
                                 int64_t ori_h, int64_t ori_w,
                                 int64_t new_h, int64_t new_w)
{
    const double scale_x = (double)ori_w / (double)new_w;
    const double scale_y = (double)ori_h / (double)new_h;

    auto flow = flow_in;
    const bool is_5d = (flow.dim() == 5);
    int64_t B = 0, T = 0, C = 0, H = 0, W = 0;
    if (is_5d) {
        B = flow.size(0); T = flow.size(1); C = flow.size(2);
        H = flow.size(3); W = flow.size(4);
        flow = flow.view({B * T, C, H, W});
    }
    flow = Fn::interpolate(flow, Fn::InterpolateFuncOptions()
               .size(std::vector<int64_t>{ori_h, ori_w})
               .mode(torch::kBilinear).align_corners(true));
    flow.index({Slice(), 0}).mul_(scale_x);   // dx
    flow.index({Slice(), 1}).mul_(scale_y);   // dy
    if (is_5d) flow = flow.view({B, T, C, ori_h, ori_w});
    return flow;
}

} // namespace megaflow
