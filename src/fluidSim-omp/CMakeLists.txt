# fluidSim-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fluidSim
    MODEL omp
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
