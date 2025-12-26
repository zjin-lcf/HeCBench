# ldpc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ldpc
    MODEL sycl
    SOURCES cpu.cpp main.cpp
    CATEGORIES algorithms
)
