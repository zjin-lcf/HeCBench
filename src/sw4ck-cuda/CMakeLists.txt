# sw4ck-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sw4ck
    MODEL cuda
    SOURCES curvilinear4sg.cu kernel1.cpp kernel2.cpp kernel3.cpp kernel4.cpp kernel5.cpp main.cu utils.cpp
    CATEGORIES algorithms
)
