# mnist-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mnist
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
