# langevin-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
