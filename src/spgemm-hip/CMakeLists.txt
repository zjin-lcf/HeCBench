# spgemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spgemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipsparse
)
