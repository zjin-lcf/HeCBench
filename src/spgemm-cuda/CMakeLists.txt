# spgemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spgemm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cusparse
)
