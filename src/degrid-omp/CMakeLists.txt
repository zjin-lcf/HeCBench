# degrid-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -DPRECISION=double
)
