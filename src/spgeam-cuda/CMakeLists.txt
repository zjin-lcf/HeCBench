# spgeam-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spgeam
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
