# lut-gemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lut-gemm
    MODEL cuda
    SOURCES kernels.cu main.cu nQWeight_fp16.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
