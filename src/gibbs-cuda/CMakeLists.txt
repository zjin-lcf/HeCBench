# gibbs-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gibbs
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
