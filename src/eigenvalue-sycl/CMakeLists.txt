# eigenvalue-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME eigenvalue
    MODEL sycl
    SOURCES kernels.cpp main.cpp reference.cpp utils.cpp
    CATEGORIES algorithms
)
