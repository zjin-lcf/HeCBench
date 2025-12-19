# determinant-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME determinant
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES cusolver
)
