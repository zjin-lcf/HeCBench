# warpsort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpsort
    MODEL sycl
    SOURCES main.cpp warpsort.cpp
    CATEGORIES algorithms
)
