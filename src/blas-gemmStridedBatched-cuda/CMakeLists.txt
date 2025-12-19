# blas-gemmStridedBatched-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blas-gemmBatched-cuda")

add_hecbench_benchmark(
    NAME blas-gemmStridedBatched
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
    LINK_LIBRARIES cublas
)
