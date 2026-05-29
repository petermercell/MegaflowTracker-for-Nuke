# ============================================================================
# CMakeLists.txt  —  MegaFlowSolve. Matches Peter's cutie_nuke build conventions:
#   libtorch from Nuke 17 (.so) + conda headers (version-matched 2.7.1),
#   -D_GLIBCXX_USE_CXX11_ABI=1, CUDA 12.8, BUILD_RPATH = Nuke + CUDA.
#   No TensorRT (pure-libtorch, all 5 engines are .pt). OpenCV for EXR + resize.
#
# Layout:
#   src/   torch_engine.{h,cpp} megaflow_solve.{h,cpp} megaflow_glue.cpp
#          megaflow_npz.cpp mf_solve.cpp  (later: MegaFlowSolve.cpp)
#   engines/   the 5 .pt  (default --engines path; not embedded)
#
# Configure / build (your A5000 box):
#   cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/mf_solve --engines engines --input .../longboard --ref_frame 25 \
#       --nuke_first_frame 1 --exr_colorspace linear --output cache/longboard_cpp.npz
# ============================================================================
cmake_minimum_required(VERSION 3.18)
project(megaflow_solve VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SRC "${CMAKE_SOURCE_DIR}/src")

# ---- Nuke 17.0 (provides libtorch 2.7.1 — link these so the plugin is ABI-
#      compatible when it loads inside Nuke; the CLI links the same set) ----
set(NUKE_ROOT "/opt/Nuke17.0v1" CACHE PATH "Nuke 17.0 install")
if(NOT EXISTS "${NUKE_ROOT}/libtorch.so")
    message(FATAL_ERROR "Nuke libtorch not found in ${NUKE_ROOT}")
endif()
set(NUKE_INCLUDE "${NUKE_ROOT}/include")

# ---- libtorch headers (conda env, version-matched to Nuke's 2.7.1) ----
#      NB: megaflow311, not cutie311.
set(TORCH_INCLUDE_ROOT
    "$ENV{HOME}/miniconda3/envs/megaflow311/lib/python3.11/site-packages/torch/include"
    CACHE PATH "libtorch headers (conda)")
if(NOT EXISTS "${TORCH_INCLUDE_ROOT}/torch/script.h")
    message(FATAL_ERROR "torch/script.h not found in ${TORCH_INCLUDE_ROOT}")
endif()
set(TORCH_INCLUDE_DIRS
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include")
set(TORCH_LIBS
    "${NUKE_ROOT}/libtorch.so" "${NUKE_ROOT}/libtorch_cpu.so"
    "${NUKE_ROOT}/libtorch_cuda.so" "${NUKE_ROOT}/libc10.so"
    "${NUKE_ROOT}/libc10_cuda.so")

# ---- CUDA 12.8 ----
if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda-12.8")
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.8")
endif()
find_package(CUDAToolkit REQUIRED)

# ---- OpenCV (CLI only: EXR decode + INTER_LINEAR resize matching cv2) ----
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# On Rocky 9, OpenCV imgcodecs pulls in libarchive.so.13, which has an
# under-satisfied libxml2 dependency (xmlTextWriter*@LIBXML2_2.6.0) at link time.
# Put the system libxml2 explicitly on the link line so those symbols resolve.
find_library(XML2_LIB NAMES xml2 libxml2.so.2 PATHS /usr/lib64 /usr/lib)
if(XML2_LIB)
    message(STATUS "  libxml2 (for libarchive symbols): ${XML2_LIB}")
else()
    message(STATUS "  libxml2 not found via find_library; will rely on -l:libxml2.so.2")
    set(XML2_LIB -l:libxml2.so.2)
endif()

# headers live in src/ (include/ also added in case they're split out later)
set(COMMON_INCLUDES
    ${SRC}
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS})

# common settings helper (mirrors cutie_common): includes, the CXX11 ABI define
# that MUST match Nuke's libtorch, warnings, -O3, and BUILD_RPATH = Nuke + CUDA.
function(mf_common tgt)
    target_include_directories(${tgt} PRIVATE ${COMMON_INCLUDES})
    target_compile_definitions(${tgt} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unused-parameter
        -Wno-deprecated-declarations -O3)
    set_target_properties(${tgt} PROPERTIES
        BUILD_RPATH "/usr/lib64;${NUKE_ROOT};${CUDAToolkit_LIBRARY_DIR}")
endfunction()

# ---- reusable solve core (shared by the CLI now, the Nuke node later) ----
add_library(megaflow_core STATIC
    ${SRC}/torch_engine.cpp
    ${SRC}/megaflow_glue.cpp
    ${SRC}/megaflow_solve.cpp
    ${SRC}/megaflow_npz.cpp)
target_link_libraries(megaflow_core PUBLIC ${TORCH_LIBS} CUDA::cudart dl rt pthread)
mf_common(megaflow_core)

# ---- standalone CLI / validation harness ----
add_executable(mf_solve ${SRC}/mf_solve.cpp)
target_link_libraries(mf_solve PRIVATE megaflow_core ${OpenCV_LIBS} ${XML2_LIB})
target_include_directories(mf_solve PRIVATE ${OpenCV_INCLUDE_DIRS})
mf_common(mf_solve)

# ---- the Nuke plugin: MegaFlowSolve.so ----
find_library(DDIMAGE_LIBRARY NAMES DDImage PATHS "${NUKE_ROOT}" NO_DEFAULT_PATH)
if(DDIMAGE_LIBRARY AND EXISTS "${SRC}/MegaFlowSolve.cpp")
    # cuda_utils.cpp compiles into the plugin (not megaflow_core / the CLI — only
    # the node has the VRAM buttons). c10_cuda (CUDACachingAllocator) + cudart it
    # needs already arrive transitively via megaflow_core's PUBLIC link.
    add_library(MegaFlowSolve SHARED
        ${SRC}/MegaFlowSolve.cpp
        ${SRC}/cuda_utils.cpp)
    # link OpenCV core+imgproc only (for cv::resize, matching the CLI's resize
    # exactly). NOT imgcodecs — the node gets pixels from Nuke, not imread, so it
    # avoids the libarchive/libxml2 chain that imgcodecs drags in.
    target_link_libraries(MegaFlowSolve PRIVATE megaflow_core ${DDIMAGE_LIBRARY}
        opencv_core opencv_imgproc)
    target_include_directories(MegaFlowSolve PRIVATE
        ${NUKE_INCLUDE} ${OpenCV_INCLUDE_DIRS})
    mf_common(MegaFlowSolve)
    set_target_properties(MegaFlowSolve PROPERTIES PREFIX "" SUFFIX ".so"
        INSTALL_RPATH "$ORIGIN:${NUKE_ROOT}")
    target_compile_options(MegaFlowSolve PRIVATE -fvisibility=hidden)
    message(STATUS "  plugin target: MegaFlowSolve.so (DDImage=${DDIMAGE_LIBRARY})")
else()
    message(STATUS "  plugin target: SKIPPED (DDImage or src/MegaFlowSolve.cpp missing)")
endif()

message(STATUS "")
message(STATUS "=== megaflow_solve === Nuke17 torch 2.7.1 (linked) | pure libtorch | CUDA ${CUDAToolkit_VERSION} | OpenCV ${OpenCV_VERSION}")
message(STATUS "  src: ${SRC}")
message(STATUS "  targets: mf_solve (CLI). MegaFlowSolve.so added after CLI validates.")
message(STATUS "======================")
