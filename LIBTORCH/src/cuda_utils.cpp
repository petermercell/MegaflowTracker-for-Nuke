// ============================================================================
// cuda_utils.cpp  —  shared libtorch VRAM helper (see cuda_utils.h).
//
// Links against the SAME libtorch the plugin uses (Nuke's bundled 2.7.1) so the
// emptyCache() call below targets the plugin's allocator. Pulls in c10_cuda
// (CUDACachingAllocator) + the CUDA runtime; both already come transitively from
// the plugin's link to megaflow_core (TORCH_LIBS + CUDA::cudart, PUBLIC).
// ============================================================================
#include "cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace cudautils {

void emptyCache() {
    c10::cuda::CUDACachingAllocator::emptyCache();
}

void reset() {
    c10::cuda::CUDACachingAllocator::emptyCache();
    cudaDeviceSynchronize();
}

MemInfo memInfo() {
    MemInfo m;
    if (cudaMemGetInfo(&m.freeBytes, &m.totalBytes) == cudaSuccess)
        m.usedBytes = m.totalBytes - m.freeBytes;
    return m;
}

std::string memInfoString() {
    MemInfo m = memInfo();
    char buf[128];
    std::snprintf(buf, sizeof(buf), "VRAM %.1f / %.1f GB used",
                  m.usedGB(), m.totalGB());
    return std::string(buf);
}

} // namespace cudautils
