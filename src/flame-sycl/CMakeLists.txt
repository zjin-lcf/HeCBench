# flame-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME flame
    MODEL sycl
    SOURCES iteration.cpp kernels.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
