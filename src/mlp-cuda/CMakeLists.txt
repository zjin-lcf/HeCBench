# mlp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES cublasLt
)
