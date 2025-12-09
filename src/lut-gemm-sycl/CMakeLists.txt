# lut-gemm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lut-gemm
    MODEL sycl
    SOURCES kernels.cpp main.cpp nQWeight_fp16.cpp
    CATEGORIES math
)
