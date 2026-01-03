# adv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -Ddfloat=double -Ddlong=int
)
