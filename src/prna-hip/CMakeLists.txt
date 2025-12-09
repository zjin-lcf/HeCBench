# prna-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME prna
    MODEL hip
    SOURCES prna.cu
    CATEGORIES algorithms
)
