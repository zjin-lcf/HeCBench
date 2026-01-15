# kmeans-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmeans
    MODEL omp
    SOURCES cluster.c getopt.c kmeans.c read_input.c rmse.c
    CATEGORIES algorithms
)
set_source_files_properties(cluster.c getopt.c kmeans.c read_input.c rmse.c PROPERTIES LANGUAGE CXX)
