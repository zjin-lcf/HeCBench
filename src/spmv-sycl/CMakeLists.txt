# spmv-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../spmv-cuda")

add_hecbench_benchmark(
    NAME spmv
    MODEL sycl
    SOURCES kernels.cpp ${INC_DIR}/utils.cpp ${INC_DIR}/main.cpp  
    CATEGORIES algorithms
    LINK_LIBRARIES -qmkl=parallel
)
