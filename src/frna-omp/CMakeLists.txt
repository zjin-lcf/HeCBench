# frna-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME frna
    MODEL omp
    SOURCES frna.cpp
    CATEGORIES algorithms
)
