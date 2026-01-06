# nosync-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nosync
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
