# frechet-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME frechet
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
