# su3-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME su3
    MODEL sycl
    SOURCES su3_nn_bench.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -DMILC_COMPLEX
)
