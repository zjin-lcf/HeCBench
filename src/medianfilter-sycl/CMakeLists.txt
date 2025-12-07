# medianfilter-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME medianfilter
    MODEL sycl
    SOURCES MedianFilter.cpp main.cpp
    CATEGORIES algorithms
)
