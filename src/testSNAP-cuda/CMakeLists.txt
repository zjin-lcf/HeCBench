# testSNAP-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME testSNAP
    MODEL cuda
    SOURCES main.cu utils.cu
    CATEGORIES algorithms
)
