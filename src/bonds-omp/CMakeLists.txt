# bonds-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bonds
    MODEL omp
    SOURCES bondsEngine.cpp bondsKernelsGpu.cpp
    CATEGORIES algorithms
)
