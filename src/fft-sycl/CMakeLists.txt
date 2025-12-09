# fft-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fft
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
