# rayleighBenardConvection-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rayleighBenardConvection
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES cublas
)
