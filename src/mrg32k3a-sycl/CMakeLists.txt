# mrg32k3a-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mrg32k3a
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES -qmkl=parallel
)
