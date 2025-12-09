# lud-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lud
    MODEL cuda
    SOURCES lud.cu lud_kernels.cu
    CATEGORIES algorithms
)
