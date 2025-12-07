# pcc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pcc
    MODEL hip
    SOURCES device.cu
    CATEGORIES algorithms
)
