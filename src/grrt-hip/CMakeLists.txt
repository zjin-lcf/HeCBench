# grrt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
