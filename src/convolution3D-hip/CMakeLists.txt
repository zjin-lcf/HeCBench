# convolution3D-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution3D
    MODEL hip
    SOURCES conv3d_s4.cu main.cu
    CATEGORIES algorithms
)
