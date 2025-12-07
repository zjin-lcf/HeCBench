# layernorm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME layernorm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
