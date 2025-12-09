# convolution3D-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution3D
    MODEL cuda
    SOURCES conv3d_s4.cu main.cu
    CATEGORIES algorithms
)
