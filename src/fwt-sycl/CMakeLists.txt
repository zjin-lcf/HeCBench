# fwt-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../fwt-cuda")

add_hecbench_benchmark(
    NAME fwt
    MODEL sycl
    SOURCES main.cpp  ${INC_DIR}/reference.cpp
    CATEGORIES algorithms
)
