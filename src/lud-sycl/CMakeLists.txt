# lud-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lud
    MODEL sycl
    SOURCES lud.cpp
    CATEGORIES algorithms
)
