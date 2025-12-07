# backprop-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME backprop
    MODEL sycl
    SOURCES backprop.cpp facetrain.cpp imagenet.cpp main.cpp
    CATEGORIES ml
)
