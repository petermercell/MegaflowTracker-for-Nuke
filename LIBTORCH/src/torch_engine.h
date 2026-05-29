// ============================================================================
// torch_engine.h  —  MegaFlow: libtorch-only TorchScript engine wrapper.
//
// Adapted from Peter's CutieRoto TorchEngine (proven). Two changes only:
//   (1) namespace cutie -> megaflow
//   (2) added a disk-load constructor (path) alongside the in-memory (ptr,size)
//       one — MegaFlow loads engines from a user-set directory (e_vit is 4.6GB;
//       embedding is out), so the file path is the primary ctor here.
// run(map<name,Tensor>, stream) is IDENTICAL to the validated version: maps
// named inputs -> positional trace args, forwards (optionally autocast fp16,
// clearing the cast-cache each call), maps positional outputs -> names.
// ============================================================================
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace megaflow {

class TorchEngine {
public:
    // Disk-load (primary for MegaFlow): load a traced .pt from `path` onto CUDA.
    TorchEngine(const std::string& path, const std::string& name,
                std::vector<std::string> inNames,
                std::vector<std::string> outNames,
                bool autocastFp16 = true);

    // In-memory load (kept for parity with the Cutie path / future embed option).
    TorchEngine(const void* data, size_t size, const std::string& name,
                std::vector<std::string> inNames,
                std::vector<std::string> outNames,
                bool autocastFp16 = true);

    // Maps inputs by name -> positional args, forwards, maps outputs back to
    // names. Ordered on the caller's stream. (Unchanged from Cutie.)
    std::map<std::string, torch::Tensor>
    run(const std::map<std::string, torch::Tensor>& inputs, cudaStream_t stream);

    const std::vector<std::string>& inputNames()  const { return inNames_; }
    const std::vector<std::string>& outputNames() const { return outNames_; }
    const std::string& name() const { return name_; }

private:
    void finishInit();   // shared post-load (eval()) for both constructors

    std::string name_;
    torch::jit::script::Module module_;
    std::vector<std::string> inNames_;
    std::vector<std::string> outNames_;
    bool autocastFp16_;
};

} // namespace megaflow
