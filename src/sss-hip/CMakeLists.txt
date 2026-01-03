# sss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL hip
    SOURCES main.cu
    INCLUDE_DIRS ${GSL_INCLUDE_DIRS}
    CATEGORIES algorithms
    LINK_LIBRARIES gsl  
)
