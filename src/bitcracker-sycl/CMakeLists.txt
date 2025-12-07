# bitcracker-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitcracker
    MODEL sycl
    SOURCES attack.cpp main.cpp utils.cpp w_blocks.cpp
    CATEGORIES algorithms
)
