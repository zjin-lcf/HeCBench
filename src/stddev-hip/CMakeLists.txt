# stddev-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME stddev
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math -munsafe-fp-atomics
)
