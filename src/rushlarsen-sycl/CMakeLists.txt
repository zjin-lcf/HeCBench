# rushlarsen-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rushlarsen
    MODEL sycl
    SOURCES kernels.cpp main.cpp reference.cpp utils.cpp
    CATEGORIES algorithms
)
