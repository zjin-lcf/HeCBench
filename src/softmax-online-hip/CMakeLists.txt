# softmax-online-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax-online
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
