# ldpc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ldpc
    MODEL omp
    SOURCES cpu.cpp main.cpp
    CATEGORIES algorithms
)
