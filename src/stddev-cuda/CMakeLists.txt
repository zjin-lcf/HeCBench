# stddev-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME stddev
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
