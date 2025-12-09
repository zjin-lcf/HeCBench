# sw4ck-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sw4ck
    MODEL sycl
    SOURCES curvilinear4sg.cpp kernel1.cpp kernel2.cpp kernel3.cpp kernel4.cpp kernel5.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
