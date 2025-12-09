# minibude-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME minibude
    MODEL sycl
    SOURCES kernel.cpp main.cpp
    CATEGORIES algorithms
)
