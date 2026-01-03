# spaxpby-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spaxpby
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES hipsparse
)
