# adv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -Ddfloat=double -Ddlong=int
)
