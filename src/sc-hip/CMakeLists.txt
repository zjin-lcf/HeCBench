# sc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sc
    MODEL hip
    SOURCES device_sc.cu main.cu
    CATEGORIES algorithms
)
