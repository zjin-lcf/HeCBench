# spsort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME spsort
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES -qmkl=parallel
)
