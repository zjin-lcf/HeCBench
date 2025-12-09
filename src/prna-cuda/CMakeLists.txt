# prna-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME prna
    MODEL cuda
    SOURCES prna.cu
    CATEGORIES algorithms
)
