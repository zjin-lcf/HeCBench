# spnnz-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spnnz
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
