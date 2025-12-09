# rng-wallace-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rng-wallace
    MODEL cuda
    SOURCES main.cu wallace_kernel.cu
    CATEGORIES algorithms
)
