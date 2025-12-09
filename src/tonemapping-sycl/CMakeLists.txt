# tonemapping-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tonemapping
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
