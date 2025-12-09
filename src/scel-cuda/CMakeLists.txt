# scel-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scel
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
