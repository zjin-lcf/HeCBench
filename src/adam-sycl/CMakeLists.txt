# adam-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME adam
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
