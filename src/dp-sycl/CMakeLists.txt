# dp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dp
    MODEL sycl
    SOURCES main.cpp shrUtils.cpp
    CATEGORIES algorithms
)
