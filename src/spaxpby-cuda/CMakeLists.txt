# spaxpby-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spaxpby
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES cusparse
)
