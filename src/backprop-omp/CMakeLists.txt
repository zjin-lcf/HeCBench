# backprop-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME backprop
    MODEL omp
    SOURCES backprop.cpp facetrain.cpp imagenet.cpp main.cpp
    CATEGORIES ml
)
