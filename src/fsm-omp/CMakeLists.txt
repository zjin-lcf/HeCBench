# fsm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fsm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
