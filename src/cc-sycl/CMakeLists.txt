# cc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME cc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
