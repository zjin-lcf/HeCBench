# softmax-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml math
)
