# threadcpy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadcpy
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
