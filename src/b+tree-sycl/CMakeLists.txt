# b+tree-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME b+tree
    MODEL sycl
    SOURCES main.c
    CATEGORIES algorithms
)
