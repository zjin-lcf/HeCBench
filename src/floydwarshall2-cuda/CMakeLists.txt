# floydwarshall2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME floydwarshall2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography graph
)
