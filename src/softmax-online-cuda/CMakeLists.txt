# softmax-online-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax-online
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
