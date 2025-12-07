# srad-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL hip
    SOURCES compress_kernel.cu extract_kernel.cu main.cu prepare_kernel.cu reduce_kernel.cu srad2_kernel.cu srad_kernel.cu
    CATEGORIES algorithms
)
