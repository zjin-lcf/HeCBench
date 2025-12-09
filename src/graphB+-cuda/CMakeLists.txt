# graphB+-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME graphB+
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
