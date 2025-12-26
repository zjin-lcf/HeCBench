# floydwarshall2-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mis-cuda")

add_hecbench_benchmark(
    NAME floydwarshall2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES graph
    INCLUDE_DIRS ${INC_DIR}
)
