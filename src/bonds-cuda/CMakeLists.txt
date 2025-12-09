# bonds-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bonds
    MODEL cuda
    SOURCES bondsEngine.cu bondsKernelsCpu.cu bondsKernelsGpu.cu
    CATEGORIES algorithms
)
