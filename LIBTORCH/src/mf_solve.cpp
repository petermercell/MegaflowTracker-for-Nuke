// ============================================================================
// mf_solve.cpp  —  standalone MegaFlow solve CLI (the validation harness).
//
// Mirrors megaflow_cache.py's IO so its .npz can be diffed against engine_cache.py
// with diff_npz.py (target ~0.006px) BEFORE any Nuke code:
//   get_native_frames  : OpenCV imread(ANYCOLOR|ANYDEPTH), BGR->RGB; float (exr)
//                        -> _float_rgb_to_uint8 (exposure, linear->sRGB, *255+0.5)
//   _to_model_tensor   : cv2.resize(uint8, (model_w,model_h), INTER_LINEAR) -> (T,3,mH,mW) float 0..255
//   build_full_trajectory / save_cache  : via megaflow_solve.{solveTrajectory,saveCache}
//
// Build: see CMakeLists.txt (links libtorch + OpenCV).
// Usage:
//   mf_solve --engines engines --input .../longboard --ref_frame 25 \
//            --nuke_first_frame 1 --exr_colorspace linear --output cache/longboard_cpp.npz
// ============================================================================
#include "megaflow_solve.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using megaflow::Engines;
using megaflow::SolveConfig;

// --- arg parsing -------------------------------------------------------------
struct Args {
    std::string engines = "engines", input, output;
    int nuke_first_frame = 1, ref_frame = -1, fix_width = 518, iters = 8;
    std::string exr_colorspace = "linear";
    float exr_exposure = 0.0f;
};
static std::string argval(int argc, char** argv, int& i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

// --- linear -> sRGB OETF (matches _linear_to_srgb) ---------------------------
static float lin2srgb(float c) {
    c = std::max(c, 0.0f);
    return (c <= 0.0031308f) ? c * 12.92f
                             : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// --- _float_rgb_to_uint8 : (H,W,3) float RGB -> uint8 ------------------------
static cv::Mat floatRgbToU8(const cv::Mat& rgbF, const std::string& cs, float exposure) {
    cv::Mat img = rgbF.clone();
    if (exposure != 0.0f) img *= std::pow(2.0f, exposure);
    cv::Mat out(img.rows, img.cols, CV_8UC3);
    for (int y = 0; y < img.rows; ++y) {
        const float* sp = img.ptr<float>(y);
        unsigned char* dp = out.ptr<unsigned char>(y);
        for (int x = 0; x < img.cols * 3; ++x) {
            float v = std::max(sp[x], 0.0f);
            if (cs == "linear") v = lin2srgb(v);
            v = std::min(std::max(v, 0.0f), 1.0f);
            dp[x] = (unsigned char)(v * 255.0f + 0.5f);   // matches *255+0.5 cast
        }
    }
    return out;   // RGB uint8
}

// --- read one image -> RGB uint8 at native res (mirrors get_native_frames) ---
static cv::Mat readNativeRGB(const std::string& path, const std::string& cs, float exposure) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    if (raw.empty()) throw std::runtime_error("cannot read image: " + path);
    if (raw.channels() == 1) cv::cvtColor(raw, raw, cv::COLOR_GRAY2BGR);
    if (raw.channels() == 4) cv::cvtColor(raw, raw, cv::COLOR_BGRA2BGR);

    cv::Mat rgb;
    cv::cvtColor(raw, rgb, cv::COLOR_BGR2RGB);    // OpenCV loads BGR

    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    const bool is_float = (raw.depth() == CV_32F || raw.depth() == CV_16F ||
                           ext == ".exr" || ext == ".hdr");
    if (is_float) {
        cv::Mat f;
        rgb.convertTo(f, CV_32FC3);               // EXR already in scene units
        return floatRgbToU8(f, cs, exposure);
    }
    cv::Mat u8;
    rgb.convertTo(u8, CV_8UC3);
    return u8;
}

static std::vector<std::string> listFrames(const std::string& dir) {
    static const std::vector<std::string> exts = {
        ".png", ".jpg", ".jpeg", ".bmp", ".exr", ".hdr", ".tif", ".tiff"};
    std::vector<std::string> out;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) != exts.end())
            out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());            // matches sorted(set(paths))
    return out;
}

// calculate_dynamic_size
static void modelSize(int orig_h, int orig_w, int fix_width, int patch,
                      int& mh, int& mw) {
    mw = fix_width;
    mh = (int)std::llround((double)orig_h * ((double)mw / (double)orig_w) / patch) * patch;
}

int main(int argc, char** argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if      (k == "--engines")          a.engines = argval(argc, argv, i);
            else if (k == "--input")            a.input = argval(argc, argv, i);
            else if (k == "--output")           a.output = argval(argc, argv, i);
            else if (k == "--nuke_first_frame") a.nuke_first_frame = std::stoi(argval(argc, argv, i));
            else if (k == "--ref_frame")        a.ref_frame = std::stoi(argval(argc, argv, i));
            else if (k == "--fix_width")        a.fix_width = std::stoi(argval(argc, argv, i));
            else if (k == "--iters")            a.iters = std::stoi(argval(argc, argv, i));
            else if (k == "--exr_colorspace")   a.exr_colorspace = argval(argc, argv, i);
            else if (k == "--exr_exposure")     a.exr_exposure = std::stof(argval(argc, argv, i));
            else throw std::runtime_error("unknown arg: " + k);
        }
        if (a.input.empty() || a.output.empty())
            throw std::runtime_error("need --input and --output");

        torch::Device dev(torch::kCUDA);
        auto stream = c10::cuda::getCurrentCUDAStream().stream();

        std::cout << "Loading engines from " << a.engines << " ...\n";
        Engines E = megaflow::loadEngines(a.engines, stream, /*autocastFp16=*/false);

        std::cout << "Reading frames from " << a.input << " ...\n";
        std::vector<std::string> paths;
        if (fs::is_directory(a.input)) paths = listFrames(a.input);
        else throw std::runtime_error("CLI v1 expects an image-sequence directory (video path: TODO)");
        if (paths.size() < 2) throw std::runtime_error("need >= 2 frames");

        // native frames -> model-res uint8 -> (T,3,mH,mW) float 0..255
        int mh = 0, mw = 0, orig_h = 0, orig_w = 0;
        std::vector<torch::Tensor> fr;
        fr.reserve(paths.size());
        for (size_t k = 0; k < paths.size(); ++k) {
            cv::Mat rgb = readNativeRGB(paths[k], a.exr_colorspace, a.exr_exposure);
            if (k == 0) {
                orig_h = rgb.rows; orig_w = rgb.cols;
                modelSize(orig_h, orig_w, a.fix_width, 14, mh, mw);
            }
            cv::Mat r;
            cv::resize(rgb, r, cv::Size(mw, mh), 0, 0, cv::INTER_LINEAR);
            // HWC uint8 -> CHW float 0..255. clone(): r.data is reused next loop,
            // and we keep these on CPU (solveTrajectory streams each pass to GPU).
            auto t = torch::from_blob(r.data, {mh, mw, 3}, torch::kUInt8)
                         .to(torch::kFloat).permute({2, 0, 1}).contiguous().clone();
            fr.push_back(t);
        }
        const int T = (int)fr.size();
        int nuke_first = a.nuke_first_frame;
        int ref_frame  = (a.ref_frame >= 0) ? a.ref_frame : nuke_first;
        int ref_idx    = ref_frame - nuke_first;
        if (ref_idx < 0 || ref_idx >= T)
            throw std::runtime_error("ref_frame outside clip range");

        auto frames_model = torch::stack(fr, 0);    // (T,3,mH,mW) CPU host RAM; solveTrajectory streams each pass to GPU
        std::cout << "  " << T << " frames at " << orig_w << "x" << orig_h
                  << " -> model " << mw << "x" << mh
                  << ", ref=" << ref_frame << " (idx " << ref_idx << ")\n";

        SolveConfig cfg;
        cfg.ref_idx = ref_idx;
        cfg.iters = a.iters;
        cfg.fix_width = a.fix_width;

        std::cout << "Running MegaFlow forward+backward ...\n";
        auto res = megaflow::solveTrajectory(E, frames_model, cfg, stream,
            [](float p){ std::cout << "  progress " << (int)(p*100) << "%\n"; return true; });

        // meta JSON — keys/values matching megaflow_cache.py exactly
        std::ostringstream m;
        m << "{"
          << "\"version\": 1, "
          << "\"input_path\": \"" << fs::absolute(a.input).string() << "\", "
          << "\"nuke_first_frame\": " << nuke_first << ", "
          << "\"last_frame\": " << (nuke_first + T - 1) << ", "
          << "\"ref_frame\": " << ref_frame << ", "
          << "\"ref_idx\": " << ref_idx << ", "
          << "\"native_h\": " << orig_h << ", "
          << "\"native_w\": " << orig_w << ", "
          << "\"model_h\": " << res.model_h << ", "
          << "\"model_w\": " << res.model_w << ", "
          << "\"fix_width\": " << a.fix_width << ", "
          << "\"iters\": " << a.iters << ", "
          << "\"dtype\": \"float16\", "
          << "\"n_frames\": " << T
          << "}";

        std::cout << "Saving cache -> " << a.output << " (float16) ...\n";
        megaflow::saveCache(a.output, res.traj_maps, m.str());
        std::cout << "Done. shape (" << res.traj_maps.size(0) << ",2,"
                  << res.model_h << "," << res.model_w << ")\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "mf_solve error: " << e.what() << "\n";
        return 1;
    }
}
