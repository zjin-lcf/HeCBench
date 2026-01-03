# sss-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES gsl  
)
