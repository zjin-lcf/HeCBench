# adv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL hip
    SOURCES adv.cu main.cu
    CATEGORIES algorithms
)
