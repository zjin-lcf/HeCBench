# ss-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ss
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
