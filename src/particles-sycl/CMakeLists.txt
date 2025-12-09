# particles-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME particles
    MODEL sycl
    SOURCES bitonicSort.cpp bitonicSort_kernels.cpp main.cpp particles.cpp particles_kernels.cpp
    CATEGORIES algorithms
)
