// ============================================================================
// cuda_utils.h  —  shared libtorch VRAM helper for Nuke plugins.
//
// Runs INSIDE the plugin's own libtorch instance (Nuke's bundled 2.7.1), so
// c10::cuda::CUDACachingAllocator::emptyCache() hits the allocator the plugin's
// tensors actually live in — unlike Python's import torch in the Script Editor,
// which loads a different libtorch and whose empty_cache() touches nothing here.
//
// Neutral namespace (cudautils::) so it can be lifted into any other libtorch
// tool later. Depends only on libtorch + the CUDA runtime — no Nuke, no OpenCV.
// ============================================================================
#pragma once

#include <cstddef>
#include <string>

namespace cudautils {

struct MemInfo {
    size_t usedBytes  = 0;
    size_t freeBytes  = 0;
    size_t totalBytes = 0;
    double usedGB()  const { return usedBytes  / 1e9; }
    double freeGB()  const { return freeBytes  / 1e9; }
    double totalGB() const { return totalBytes / 1e9; }
};

// Frees cached-but-unused blocks held by libtorch's CUDA caching allocator.
// Runs inside the plugin's libtorch instance -> targets the right allocator.
void emptyCache();

// Empty cache + device synchronize, for a hard reclaim.
void reset();

// Current device VRAM usage (driver-level, via cudaMemGetInfo).
MemInfo memInfo();

// Convenience one-liner, e.g. "VRAM 9.2 / 24.0 GB used".
std::string memInfoString();

} // namespace cudautils
