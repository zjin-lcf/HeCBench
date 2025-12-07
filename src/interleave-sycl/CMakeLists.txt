# interleave-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME interleave
    MODEL sycl
    SOURCES main.cpp util.cpp
    CATEGORIES algorithms
)
