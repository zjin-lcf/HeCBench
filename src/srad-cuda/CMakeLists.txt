# srad-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
