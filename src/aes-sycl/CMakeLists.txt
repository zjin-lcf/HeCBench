# aes-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME aes
    MODEL sycl
    SOURCES kernels.cpp main.cpp reference.cpp utils.cpp
    CATEGORIES cryptography
)
