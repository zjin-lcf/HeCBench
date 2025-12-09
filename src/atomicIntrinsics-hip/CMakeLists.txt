# atomicIntrinsics-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicIntrinsics
    MODEL hip
    SOURCES main-um.cu main.cu
    CATEGORIES algorithms
)
