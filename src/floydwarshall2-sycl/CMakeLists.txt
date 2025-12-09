# floydwarshall2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME floydwarshall2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES cryptography graph
)
