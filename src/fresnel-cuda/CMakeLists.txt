# fresnel-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fresnel
    MODEL cuda
    SOURCES cosine.cu fresnel.cu main.cu sine.cu xchebyshev.cu
    CATEGORIES algorithms
)
