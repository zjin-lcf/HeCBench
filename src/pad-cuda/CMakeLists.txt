# pad-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pad
    MODEL cuda
    SOURCES host_task.cpp kernel.cu main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
