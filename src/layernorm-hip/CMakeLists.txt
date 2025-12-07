# layernorm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME layernorm
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
