# sobel-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sobel
    MODEL sycl
    SOURCES kernels.cpp main.cpp reference.cpp
    CATEGORIES algorithms
)
