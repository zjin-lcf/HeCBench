# testSNAP-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME testSNAP
    MODEL sycl
    SOURCES main.cpp utils.cpp
    CATEGORIES algorithms
)
