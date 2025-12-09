# rsc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsc
    MODEL sycl
    SOURCES main.cpp model_eval.cpp
    CATEGORIES algorithms
)
