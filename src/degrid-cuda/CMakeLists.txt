# degrid-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
