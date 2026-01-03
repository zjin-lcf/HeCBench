# atomicCost-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCost
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -munsafe-fp-atomics
)
