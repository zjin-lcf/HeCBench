# hogbom-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hogbom
    MODEL sycl
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
