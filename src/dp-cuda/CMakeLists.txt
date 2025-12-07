# dp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dp
    MODEL cuda
    SOURCES main.cu shrUtils.cu
    CATEGORIES algorithms
)
