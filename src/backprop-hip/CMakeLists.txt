# backprop-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME backprop
    MODEL hip
    SOURCES backprop.cu facetrain.cu imagenet.cu main.cu
    CATEGORIES ml
)
