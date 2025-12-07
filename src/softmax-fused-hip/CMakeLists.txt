# softmax-fused-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax-fused
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
