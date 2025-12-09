# daphne-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME daphne
    MODEL sycl
    SOURCES
        src/common/main.cpp
        src/points2image/kernel.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    CATEGORIES imaging automotive
)
