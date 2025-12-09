# rsc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsc
    MODEL omp
    SOURCES main.cpp model_eval.cpp
    CATEGORIES algorithms
)
