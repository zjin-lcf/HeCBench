# gabor-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gabor
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
