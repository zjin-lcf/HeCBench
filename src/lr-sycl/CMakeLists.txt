# lr-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lr
    MODEL sycl
    SOURCES linear_iter.cpp linear_par.cpp main.cpp
    CATEGORIES algorithms
)
