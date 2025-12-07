# perplexity-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME perplexity
    MODEL cuda
    SOURCES main.cu reference.cpp
    CATEGORIES algorithms
)
