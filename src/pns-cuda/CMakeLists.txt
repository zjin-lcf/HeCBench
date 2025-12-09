# pns-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pns
    MODEL cuda
    SOURCES main.cu petri_kernel.cu rand_gen.cu
    CATEGORIES algorithms
)
