# minisweep-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME minisweep
    MODEL sycl
    SOURCES kernels.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
