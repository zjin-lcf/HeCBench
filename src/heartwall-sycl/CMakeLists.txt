# heartwall-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME heartwall
    MODEL sycl
    SOURCES main.c
    CATEGORIES algorithms
)
