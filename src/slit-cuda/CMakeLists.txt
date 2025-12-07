# slit-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME slit
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
