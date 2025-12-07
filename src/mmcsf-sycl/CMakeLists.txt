# mmcsf-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mmcsf
    MODEL sycl
    SOURCES kernels.cpp main.cpp mttkrp_cpu.cpp
    CATEGORIES algorithms
)
