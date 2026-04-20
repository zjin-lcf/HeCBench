# mnist-sycl/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../mnist-cuda/data.zip")

add_hecbench_benchmark(
    NAME mnist
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
