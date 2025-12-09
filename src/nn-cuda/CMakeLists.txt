# nn-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nn
    MODEL cuda
    SOURCES nearestNeighbor.cu utils.cu
    CATEGORIES algorithms
)
