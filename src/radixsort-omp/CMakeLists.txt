# radixsort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort
    MODEL omp
    SOURCES RadixSort.cpp RadixSort_kernels.cpp Scan.cpp Scan_kernels.cpp main.cpp
    CATEGORIES algorithms
)
