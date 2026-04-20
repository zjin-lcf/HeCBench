# clink-sycl/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.bz")

add_hecbench_benchmark(
    NAME clink
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
