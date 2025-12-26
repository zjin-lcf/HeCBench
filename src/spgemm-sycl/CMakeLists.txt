# spgemm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME spgemm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
