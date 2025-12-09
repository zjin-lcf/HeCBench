# frna-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME frna
    MODEL sycl
    SOURCES frna.cpp
    CATEGORIES algorithms
)
