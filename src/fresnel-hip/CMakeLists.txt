# fresnel-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fresnel
    MODEL hip
    SOURCES cosine.cu fresnel.cu main.cu sine.cu xchebyshev.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -fgpu-rdc -ffast-math
    LINK_LIBRARIES -fgpu-rdc --hip-link
)
