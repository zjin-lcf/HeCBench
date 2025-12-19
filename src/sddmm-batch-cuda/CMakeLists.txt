# sddmm-batch-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sddmm-batch
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES cusparse
)
