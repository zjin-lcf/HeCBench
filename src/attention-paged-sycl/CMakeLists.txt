# attention-paged-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME attention-paged
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
