# aes-omp/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME aes
    MODEL omp
    SOURCES main.cpp
    CATEGORIES cryptography
    INCLUDE_DIRS ${SDK_DIR}
)
