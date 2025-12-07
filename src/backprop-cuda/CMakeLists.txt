# backprop-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME backprop
    MODEL cuda
    SOURCES backprop.cu facetrain.cu imagenet.cu main.cu
    CATEGORIES ml
)
