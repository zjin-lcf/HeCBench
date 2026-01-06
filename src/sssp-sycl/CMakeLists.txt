# sssp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sssp
    MODEL sycl
    SOURCES kernel.cpp main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
