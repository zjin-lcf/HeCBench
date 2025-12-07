# pns-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pns
    MODEL omp
    SOURCES main.cpp petri_kernel.cpp rand_gen.cpp
    CATEGORIES algorithms
)
