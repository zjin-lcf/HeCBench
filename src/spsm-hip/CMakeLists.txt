# spsm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spsm
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES hipsparse
)
