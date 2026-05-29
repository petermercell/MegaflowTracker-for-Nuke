// ============================================================================
// torch_engine.cpp  —  MegaFlow libtorch-only engine wrapper (see header).
// Adapted from Peter's validated CutieRoto TorchEngine.
//
// ONE DELIBERATE DIFFERENCE FROM CUTIE, and it matters for matching the oracle:
//   Cutie's run() did `.to(kFloat32)` on every output (its memory core is fp32).
//   MegaFlow must NOT — engine_cache.py's run_engine returns the engine's NATIVE
//   dtype (fp16 from the fp16-traced engines) and hands it straight to the next
//   engine / the glue under one OUTER autocast. Casting to fp32 between engines
//   would change the inter-stage precision and diverge from the .npz we validated
//   (ab_cache: 0.006px). So outputs are returned as-is.
//
// AUTOCAST: the solve core sets ONE bf16 autocast scope around the whole windowed
// solve (mirroring build_full_trajectory's `torch.autocast('cuda', bf16)`), and
// calls engines with autocastFp16=false so they inherit that scope — exactly like
// engine_cache.py's run_engine (which adds no autocast of its own). The
// autocastFp16=true path is kept only for the standalone single-engine unit-diff.
// ============================================================================
#include "torch_engine.h"

#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <sstream>

namespace megaflow {

void TorchEngine::finishInit()
{
    module_.eval();
}

TorchEngine::TorchEngine(const std::string& path, const std::string& name,
                         std::vector<std::string> inNames,
                         std::vector<std::string> outNames,
                         bool autocastFp16)
    : name_(name),
      inNames_(std::move(inNames)),
      outNames_(std::move(outNames)),
      autocastFp16_(autocastFp16)
{
    try {
        module_ = torch::jit::load(path, torch::kCUDA);
    } catch (const std::exception& e) {
        throw std::runtime_error("TorchEngine[" + name_ + "] load failed from '" +
                                 path + "': " + e.what());
    }
    finishInit();
}

TorchEngine::TorchEngine(const void* data, size_t size, const std::string& name,
                         std::vector<std::string> inNames,
                         std::vector<std::string> outNames,
                         bool autocastFp16)
    : name_(name),
      inNames_(std::move(inNames)),
      outNames_(std::move(outNames)),
      autocastFp16_(autocastFp16)
{
    std::string blob(reinterpret_cast<const char*>(data), size);
    std::istringstream iss(blob, std::ios::binary);
    try {
        module_ = torch::jit::load(iss, torch::kCUDA);
    } catch (const std::exception& e) {
        throw std::runtime_error("TorchEngine[" + name_ + "] load failed: " + e.what());
    }
    finishInit();
}

std::map<std::string, torch::Tensor>
TorchEngine::run(const std::map<std::string, torch::Tensor>& inputs, cudaStream_t stream)
{
    torch::NoGradGuard ng;
    auto s = c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
    c10::cuda::CUDAStreamGuard guard(s);

    std::vector<torch::jit::IValue> args;
    args.reserve(inNames_.size());
    for (const auto& n : inNames_) {
        auto it = inputs.find(n);
        if (it == inputs.end())
            throw std::runtime_error("TorchEngine[" + name_ + "] missing input '" + n + "'");
        args.emplace_back(it->second);
    }

    torch::jit::IValue out;
    if (autocastFp16_) {
        // standalone unit-diff mode only: self-contained fp16 scope + cache clear.
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        out = module_.forward(args);
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::clear_cache();
    } else {
        // solve-core mode: inherit the outer bf16 autocast scope (like run_engine).
        out = module_.forward(args);
    }

    std::vector<torch::Tensor> outs;
    if (out.isTensor()) {
        outs.push_back(out.toTensor());
    } else if (out.isTuple()) {
        for (const auto& e : out.toTuple()->elements())
            outs.push_back(e.toTensor());
    } else if (out.isTensorList()) {
        for (const auto& e : out.toTensorList())
            outs.push_back(e);
    } else if (out.isList()) {
        for (const auto& e : out.toList())
            outs.push_back(e.get().toTensor());
    } else {
        throw std::runtime_error("TorchEngine[" + name_ + "] unexpected output type: " +
                                 out.tagKind());
    }
    if (outs.size() != outNames_.size())
        throw std::runtime_error("TorchEngine[" + name_ + "] output count " +
            std::to_string(outs.size()) + " != names " + std::to_string(outNames_.size()));

    std::map<std::string, torch::Tensor> result;
    for (size_t i = 0; i < outNames_.size(); ++i)
        result[outNames_[i]] = outs[i];     // native dtype — NO fp32 cast (see header note)
    return result;
}

} // namespace megaflow
