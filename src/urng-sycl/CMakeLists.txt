# urng-sycl/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME urng
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    INCLUDE_DIRS ${SDK_DIR}
)
