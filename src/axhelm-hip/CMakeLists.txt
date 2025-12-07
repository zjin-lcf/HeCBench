# axhelm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME axhelm
    MODEL hip
    SOURCES axhelmKernel.cpp main.cu meshBasis.cpp
    CATEGORIES algorithms
)
