# tpacf-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tpacf
    MODEL cuda
    SOURCES ACF_kernel.cu compute.cu histogram_kernel.cu model_io.cu
    CATEGORIES algorithms
)
