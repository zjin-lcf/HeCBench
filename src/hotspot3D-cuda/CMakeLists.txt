# hotspot3D-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hotspot3D
    MODEL cuda
    SOURCES 3D.cu 3D_helper.cu
    CATEGORIES algorithms
)
