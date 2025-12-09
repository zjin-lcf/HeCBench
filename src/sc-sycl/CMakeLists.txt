# sc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sc
    MODEL sycl
    SOURCES device_sc.cpp main.cpp
    CATEGORIES algorithms
)
