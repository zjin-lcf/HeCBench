# btree-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME btree
    MODEL sycl
    SOURCES host.cpp main.cpp
    CATEGORIES algorithms
)
