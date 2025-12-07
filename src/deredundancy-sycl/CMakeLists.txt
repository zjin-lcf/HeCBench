# deredundancy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL sycl
    SOURCES kernels.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
