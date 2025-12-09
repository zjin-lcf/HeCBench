# pcc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pcc
    MODEL sycl
    SOURCES device.cpp
    CATEGORIES algorithms
)
