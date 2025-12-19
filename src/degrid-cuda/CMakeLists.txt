# degrid-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -DPRECISION=double
)
