# prna-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME prna
    MODEL omp
    SOURCES prna.cpp main.c base.c param.c util.c
    CATEGORIES algorithms
    COMPILE_OPTIONS -DOMP_TARGET -DFLOAT
    ENABLE_FASTMATH true
)
