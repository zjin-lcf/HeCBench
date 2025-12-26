# streamUM-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamUM
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -fopenmp 
    LINK_LIBRARIES -qmkl=parallel -fopenmp
)
