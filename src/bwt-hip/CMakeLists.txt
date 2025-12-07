# bwt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bwt
    MODEL hip
    SOURCES bwt.cu
    CATEGORIES algorithms
)
