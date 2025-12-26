# rayleighBenardConvection-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rayleighBenardConvection
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES -qmkl=parallel
)
