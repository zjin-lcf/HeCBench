# convolution3D-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution3D
    MODEL sycl
    SOURCES conv3d_s4.cpp main.cpp
    CATEGORIES algorithms
)
