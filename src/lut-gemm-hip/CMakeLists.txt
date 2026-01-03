# lut-gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lut-gemm
    MODEL hip
    SOURCES kernels.cu main.cu nQWeight_fp16.cu
    CATEGORIES math
    COMPILE_OPTIONS -DHIPBLAS_V2
    LINK_LIBRARIES hipblas
)
