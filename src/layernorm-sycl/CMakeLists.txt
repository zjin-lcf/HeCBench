# layernorm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME layernorm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
