# prna-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME prna
    MODEL omp
    SOURCES prna.cpp
    CATEGORIES algorithms
)
