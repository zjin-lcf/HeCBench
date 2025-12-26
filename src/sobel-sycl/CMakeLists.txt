# sobel-sycl/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME sobel
    MODEL sycl
    SOURCES main.cpp reference.cpp
    CATEGORIES algorithms
    INCLUDE_DIRS ${SDK_DIR}
)
