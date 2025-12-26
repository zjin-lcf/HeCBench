# blas-gemmStridedBatched-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blas-gemmBatched-cuda")

add_hecbench_benchmark(
    NAME blas-gemmStridedBatched
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
    LINK_LIBRARIES -qmkl=parallel
)
