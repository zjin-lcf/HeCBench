# pcc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pcc
    MODEL cuda
    SOURCES device.cu main.cpp
    CATEGORIES algorithms
)
