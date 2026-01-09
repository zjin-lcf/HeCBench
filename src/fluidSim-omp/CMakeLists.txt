# fluidSim-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fluidSim
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -DVERIFY
)
