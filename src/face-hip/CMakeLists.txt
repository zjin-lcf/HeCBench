# face-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME face
    MODEL hip
    SOURCES haar.cu
    CATEGORIES algorithms
)
