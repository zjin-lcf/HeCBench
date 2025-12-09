# cfd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cfd
    MODEL cuda
    SOURCES euler3d.cu
    CATEGORIES algorithms
)
