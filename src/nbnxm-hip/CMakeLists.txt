# nbnxm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nbnxm
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -munsafe-fp-atomics
)
