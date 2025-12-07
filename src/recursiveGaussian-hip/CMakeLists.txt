# recursiveGaussian-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME recursiveGaussian
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
