# qkv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME qkv
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES hipblas hipblaslt
)
