# aes-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME aes
    MODEL omp
    SOURCES kernels.cpp main.cpp reference.cpp utils.cpp
    CATEGORIES cryptography
)
