# local-ht-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME local-ht
    MODEL cuda
    SOURCES
        src/main.cu
        src/kernel.cu
        src/helper.cu
    CATEGORIES bioinformatics
)
