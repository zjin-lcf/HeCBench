# pso-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pso
    MODEL sycl
    SOURCES kernel_gpu.cpp
    CATEGORIES algorithms
)
