# eikonal-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME eikonal
    MODEL hip
    SOURCES StructuredEikonal.cu fim.cu kernel.cu
    CATEGORIES algorithms
)
