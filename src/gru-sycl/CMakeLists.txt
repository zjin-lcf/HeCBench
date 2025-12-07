# gru-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gru
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
