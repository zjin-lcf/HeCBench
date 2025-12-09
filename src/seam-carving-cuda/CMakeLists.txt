# seam-carving-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME seam-carving
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
