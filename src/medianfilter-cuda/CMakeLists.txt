# medianfilter-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME medianfilter
    MODEL cuda
    SOURCES MedianFilter.cu MedianFilterHost.cpp main.cu
    CATEGORIES algorithms
)
