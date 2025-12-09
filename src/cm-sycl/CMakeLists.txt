# cm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME cm
    MODEL sycl
    SOURCES main.cpp process.cpp
    CATEGORIES algorithms
)
