# srad-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
