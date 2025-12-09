# bonds-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bonds
    MODEL sycl
    SOURCES bondsEngine.cpp bondsKernelsCpu.cpp bondsKernelsGpu.cpp
    CATEGORIES algorithms
)
