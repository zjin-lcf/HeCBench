# mlp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlp
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES hipblaslt
)
