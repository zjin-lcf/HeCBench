# frna-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME frna
    MODEL hip
    SOURCES frna.cu
    CATEGORIES algorithms
)
