# wordcount-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME wordcount
    MODEL cuda
    SOURCES wc.cu
    CATEGORIES algorithms
)
