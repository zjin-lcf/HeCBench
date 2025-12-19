# frechet-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME frechet
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
