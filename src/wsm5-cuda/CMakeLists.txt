# wsm5-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME wsm5
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -DMKX=4
)
