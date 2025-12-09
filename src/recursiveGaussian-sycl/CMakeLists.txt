# recursiveGaussian-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME recursiveGaussian
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
