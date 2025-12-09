# gmm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gmm
    MODEL sycl
    SOURCES cluster.cpp gaussian_kernel.cpp main.cpp readData.cpp
    CATEGORIES algorithms
)
