# unfold-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME unfold
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS --extended-lambda
)
