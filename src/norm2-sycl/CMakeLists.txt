# norm2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME norm2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES -qmkl=parallel
)
