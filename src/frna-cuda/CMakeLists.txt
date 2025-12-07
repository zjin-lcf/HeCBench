# frna-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME frna
    MODEL cuda
    SOURCES frna.cu
    CATEGORIES algorithms
)
