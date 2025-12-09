# spmv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME spmv
    MODEL sycl
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
