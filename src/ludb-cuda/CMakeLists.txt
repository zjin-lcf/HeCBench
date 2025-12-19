# ludb-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ludb
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES cublas
)
