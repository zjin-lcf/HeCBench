# tpacf-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tpacf
    MODEL hip
    SOURCES ACF_kernel.cu compute.cu histogram_kernel.cu
    CATEGORIES algorithms
)
