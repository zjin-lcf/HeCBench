# tridiagonal-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tridiagonal
    MODEL cuda
    SOURCES cmd_arg_reader.cu cyclic_kernels.cu main.cu pcr_kernels.cu shrUtils.cu sweep_kernels.cu
    CATEGORIES algorithms
)
