# blas-gemmStridedBatched-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blas-gemmBatched-cuda")

add_hecbench_benchmark(
    NAME blas-gemmStridedBatched
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    COMPILE_OPTIONS -DHIPBLAS_USE_HIP_HALF
    INCLUDE_DIRS ${INC_DIR}
    LINK_LIBRARIES hipblas
)
