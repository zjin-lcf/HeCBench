# lut-gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lut-gemm
    MODEL hip
    SOURCES kernels.cu main.cu nQWeight_fp16.cu
    CATEGORIES math
)
