# adamw-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME adamw
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    COMPILE_OPTIONS -std=c++20
)
