# bicgstab-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bicgstab
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
