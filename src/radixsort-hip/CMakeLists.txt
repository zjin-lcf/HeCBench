# radixsort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort
    MODEL hip
    SOURCES RadixSort.cu RadixSort_kernels.cu Scan.cu Scan_kernels.cu main.cu
    CATEGORIES algorithms
)
