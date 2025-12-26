# fresnel-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fresnel
    MODEL sycl
    SOURCES cosine.cpp fresnel.cpp main.cpp sine.cpp xchebyshev.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
