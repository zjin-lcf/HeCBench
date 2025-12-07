# mmcsf-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mmcsf
    MODEL cuda
    SOURCES kernels.cu main.cu mttkrp_cpu.cu
    CATEGORIES algorithms
)
