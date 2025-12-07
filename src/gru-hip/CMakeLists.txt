# gru-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gru
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
