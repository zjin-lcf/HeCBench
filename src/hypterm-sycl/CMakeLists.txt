# hypterm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hypterm
    MODEL sycl
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
