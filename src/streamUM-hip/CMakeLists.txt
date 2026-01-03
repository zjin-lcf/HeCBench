# streamUM-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamUM
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -fopenmp 
    LINK_LIBRARIES hipblas -fopenmp
)
