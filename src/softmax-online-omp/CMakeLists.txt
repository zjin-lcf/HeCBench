# softmax-online-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax-online
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
