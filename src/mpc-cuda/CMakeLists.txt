# mpc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mpc
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
