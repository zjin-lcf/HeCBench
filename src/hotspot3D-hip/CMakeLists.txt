# hotspot3D-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hotspot3D
    MODEL hip
    SOURCES 3D.cu 3D_helper.cu
    CATEGORIES algorithms
)
