# simpleSpmv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME simpleSpmv
    MODEL hip
    SOURCES kernels.cu
    CATEGORIES algorithms
)
