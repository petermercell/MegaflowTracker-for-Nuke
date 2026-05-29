// ============================================================================
// megaflow_solve.h  —  MegaFlow native solve core (libtorch-only).
//
// Pure C++/libtorch transcription of the validated Python pipeline:
//   engine_cache.py  -> build_full_trajectory (megaflow_cache.py)
//                    -> model.forward_track   (megaflow.py, windowed)
//                    -> glue: global_correlation_softmax / local_correlation_with_flow
//                             / coords_grid / upsample_flow_with_mask / resize_flow_bilinear
//
// SCOPE: this core ONLY produces the trajectory .npz ref-cache. Picking and
// Tracker/CornerPin baking stay in Peter's existing, shipping gizmo. The Nuke
// node is a Solve-button wrapper around solveTrajectory(); the same core also
// drives a standalone CLI (mf_solve) used to diff against the Python oracle
// (diff_npz.py / ab_cache.py) BEFORE any Nuke code runs.
//
// Precision: engines traced fp16 (validated: 0.046px worst-fwd vs model, 0.006px
// overall median, multi-window 51-frame EXR). Glue runs under the same autocast
// regime the Python build used (bf16 outer / fp16 engines is what the .npz the
// gizmo reads was validated against). The solve core mirrors that exactly.
// ============================================================================
#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "torch_engine.h"   // megaflow::TorchEngine (libtorch-only, disk-load ctor)

namespace megaflow {

// ---------------------------------------------------------------------------
// Engine set. Loaded once from a directory of 5 .pt files (NOT embedded — e_vit
// is 4.6GB). Path resolved by the caller (knob / MEGAFLOW_ENGINES env / default).
// IO names match export_engines.py's printed wiring exactly.
// ---------------------------------------------------------------------------
struct Engines {
    std::unique_ptr<TorchEngine> vit;     // {img}                      -> {tokens}
    std::unique_ptr<TorchEngine> cnn;     // {frames}                   -> {cnn0,cnn1}
    std::unique_ptr<TorchEngine> head;    // {tokens,img,cnn0,cnn1}     -> {feat}
    std::unique_ptr<TorchEngine> proj;    // {refine_feat}              -> {proj}
    std::unique_ptr<TorchEngine> refine;  // {net,inp,corr,flow} -> {net,up_masks,residual_flows}

    // model hyperparams read from the trace / fixed by export (S=16, patch=14,
    // refine_factor=4, feat_ch=128, patch_start_idx=4). Filled by loadEngines.
    int seq_len = 16, patch_size = 14, refine_factor = 4,
        feature_channels = 128, patch_start_idx = 4;
};

// Load all 5 .pt from `dir`. Throws std::runtime_error naming the missing/failed
// file. autocastFp16 matches the engines' trace precision.
Engines loadEngines(const std::string& dir, cudaStream_t stream,
                    bool autocastFp16 = true);

// ---------------------------------------------------------------------------
// Solve configuration — mirrors engine_cache.py CLI args.
// ---------------------------------------------------------------------------
struct SolveConfig {
    int ref_idx     = 0;     // 0-based reference frame within the stack
    int iters       = 8;     // num_reg_refine (refine iterations per window)
    int fix_width   = 518;   // model resize target width (calculate_dynamic_size)
    // solver is always forward_track (global anchor; validated). sliding dropped.
};

struct SolveResult {
    torch::Tensor traj_maps;     // (T,2,Hm,Wm) CPU float — ABSOLUTE coords, model space
    int model_h = 0, model_w = 0;
};

// progress: called with [0,1]; return false to cancel (wired to Nuke ProgressTask
// in the node, no-op in the CLI). Optional.
using ProgressFn = std::function<bool(float)>;

// The full solve = build_full_trajectory. `frames_model`: (T,3,model_H,model_W)
// float CUDA tensor, RGB, values 0..255 (the caller does the cv2 native->model
// INTER_LINEAR resize + EXR->uint8 encode, matching _to_model_tensor exactly).
// Runs forward (ref..T) AND backward (ref..0) passes via runMegaflow, flips the
// backward half, concatenates, forces identity at ref. Returns traj_maps
// (T,2,model_H,model_W) absolute coords in MODEL space on CPU.
SolveResult solveTrajectory(Engines& E, const torch::Tensor& frames_model,
                            const SolveConfig& cfg, cudaStream_t stream,
                            ProgressFn progress = {});

// ---------------------------------------------------------------------------
// Windowed forward_track (transcribed from megaflow.py). Exposed so the CLI can
// diff a single forward pass against reference_solve_window.py directly.
//   imgs_model: (1,S,3,Hm,Wm) padded to /8. Returns flow_final (1,S,2,Hm,Wm).
// ---------------------------------------------------------------------------
torch::Tensor forwardTrack(Engines& E, const torch::Tensor& imgs_model,
                           int num_reg_refine, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Pure-torch glue (transcribed from matching.py / model_utils.py / geometry.py).
// Each exposed for standalone unit-diff against the Python function on captured
// tensors — same discipline as the per-stage engine checks.
// ---------------------------------------------------------------------------
torch::Tensor globalCorrelationSoftmax(const torch::Tensor& feat0,
                                       const torch::Tensor& feat1);     // -> flow (B,2,H,W)
torch::Tensor localCorrelationWithFlow(const torch::Tensor& f0,
                                       const torch::Tensor& f1,
                                       const torch::Tensor& flow,
                                       int local_radius = 4);            // -> corr (B,(2r+1)^2,H,W)
torch::Tensor coordsGrid(int64_t b, int64_t h, int64_t w, torch::Device dev);
torch::Tensor upsampleFlowWithMask(const torch::Tensor& flow,
                                   const torch::Tensor& mask, int factor);
torch::Tensor resizeFlowBilinear(const torch::Tensor& flow,
                                 int64_t ori_h, int64_t ori_w,
                                 int64_t new_h, int64_t new_w);

// ---------------------------------------------------------------------------
// .npz writer — must match Peter's save_cache schema byte-for-readable:
//   traj_maps : (T,2,Hm,Wm) float16   + meta : JSON (ref_idx, model_h/w, n_frames,
//   native_h/w, nuke_first_frame, last_frame, ref_frame, fix_width, iters, dtype...)
// (exact key names / meta storage form pending megaflow_cache.py::save_cache source)
// ---------------------------------------------------------------------------
void saveCache(const std::string& path,
               const torch::Tensor& traj_maps,      // (T,2,Hm,Wm), cast to fp16 inside
               const std::string& meta_json);

} // namespace megaflow
