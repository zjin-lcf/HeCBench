# recursiveGaussian-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME recursiveGaussian
    MODEL cuda
    SOURCES RecursiveGaussianHost.cpp main.cu
    CATEGORIES algorithms
)
