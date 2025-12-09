# degrid-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
