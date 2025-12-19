# meanshift-cuda/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/dataset.tar.gz")

add_hecbench_benchmark(
    NAME meanshift
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -use_fast_math
)
