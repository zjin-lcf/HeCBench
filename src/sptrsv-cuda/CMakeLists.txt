# sptrsv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sptrsv
    MODEL cuda
    SOURCES sptrsv_syncfree.cu
    CATEGORIES algorithms
)
