// ============================================================================
// MegaFlowSolve.h  —  the Nuke node (Phase 3). A Solve-button Iop that wraps the
// validated megaflow::solveTrajectory core and writes the .npz ref-cache.
//
// SCOPE: this node ONLY generates the trajectory .npz that Peter's existing,
// shipping gizmo (MEGAFlowTracker_PM) reads via its `npz` File_Knob. Picking and
// Tracker4/CornerPin baking stay entirely in that gizmo. The two are independent
// nodes that share a file path — no coupling.
//
// Design (mirrors CutieRoto's button/pull/progress conventions):
//   * input 0 = plate (required). Passthrough on the image pin — engine() just
//     pipes input(0) through; the node is NOT a per-frame processor.
//   * a Solve button runs the whole forward+backward solve once via inputAt()
//     frame-pulls (like processAllFrames), under a progress task, then writes npz.
//   * frames are pulled native-res RGB (bottom-up -> top-down flip, exactly like
//     pullInputResized), encoded to 0..255 matching _float_rgb_to_uint8, resized
//     to model res, stacked (T,3,mH,mW) and handed to solveTrajectory.
//   * engines loaded once from a dir knob (or MEGAFLOW_ENGINES env), libtorch
//     stream from libtorch's own pool (Windows one-runtime safety, like Cutie).
//
// VALIDATION: the node writes the SAME npz the CLI (mf_solve) does. The CLI's
// output diffs 0.0000px against the Python oracle (engine_cache.py). So the node
// is validated the same way: Solve a plate, diff_npz its npz vs the CLI's.
// ============================================================================
#pragma once

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Knob.h"
#include "DDImage/Format.h"
#include "DDImage/OutputContext.h"

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <memory>

#include "megaflow_solve.h"   // megaflow::Engines / SolveConfig / solveTrajectory / saveCache

using namespace DD::Image;

class MegaFlowSolve : public Iop {
public:
    MegaFlowSolve(Node* node);
    ~MegaFlowSolve() override;

    const char* Class() const override { return description.name; }
    const char* node_help() const override;

    // input 0 = plate (required). Single input.
    int minimum_inputs() const override { return 1; }
    int maximum_inputs() const override { return 1; }
    bool test_input(int n, Op* op) const override;
    const char* input_label(int n, char* buf) const override;

    void knobs(Knob_Callback f) override;
    int  knob_changed(Knob* k) override;

    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t, ChannelMask m, int count) override;
    void engine(int y, int x, int r, ChannelMask m, Row& row) override;

    static const Iop::Description description;

private:
    // ---- knobs ----
    const char* enginesDirKnob_;   // dir of the 5 .pt (else MEGAFLOW_ENGINES env)
    const char* npzPathKnob_;      // output .npz (point the gizmo's `npz` knob here)
    int   refFrame_;               // reference frame in NUKE frame numbers
    int   firstFrame_;             // 0 = use input's first frame
    int   lastFrame_;              // 0 = use input's last frame
    int   iters_;                  // num_reg_refine (default 8)
    int   fixWidth_;               // model resize target width (default 518)
    int   colorspace_;             // 0 = linear (apply lin->sRGB), 1 = sRGB/raw (as-is)
    int   gpuDevice_;

    // ---- engines (loaded once, on first Solve) ----
    std::unique_ptr<megaflow::Engines> engines_;
    cudaStream_t stream_ = nullptr;
    bool enginesReady_ = false;
    std::string loadedFrom_;       // dir the engines were loaded from (reload if changed)

    // ---- plate geometry (input 0) ----
    int fX_ = 0, fY_ = 0, fW_ = 0, fH_ = 0;

    // ---- helpers ----
    std::string resolveEnginesDir() const;     // knob, else $MEGAFLOW_ENGINES
    void ensureEngines();                      // build stream + loadEngines once
    bool inputFrameRange(int& first, int& last) const;  // input 0's frame range

    // pull input 0 at `frame` -> native-res RGB uint8 cv::Mat (HWC, top-down),
    // identical to the CLI's get_native_frames + _float_rgb_to_uint8. The cv::resize
    // to model res happens in doSolve, matching the CLI byte-for-byte.
    cv::Mat pullNativeRGB255(double frame, int& nativeH, int& nativeW);

    void doSolve();                            // the Solve button action
};
