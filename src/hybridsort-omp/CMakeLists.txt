# hybridsort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hybridsort
    MODEL omp
    SOURCES bucketsort.c hybridsort.c mergesort.c
    CATEGORIES algorithms
)
set_source_files_properties(bucketsort.c hybridsort.c mergesort.c PROPERTIES LANGUAGE CXX)
