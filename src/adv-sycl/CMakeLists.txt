# adv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
