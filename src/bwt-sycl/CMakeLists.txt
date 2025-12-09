# bwt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bwt
    MODEL sycl
    SOURCES bwt.cpp
    CATEGORIES algorithms
)
