# logan-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME logan
    MODEL sycl
    SOURCES
        src/main.cpp
        src/seed.cpp
        src/score.cpp
        src/logan_functions.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    COMPILE_OPTIONS -DADAPTABLE
    CATEGORIES bioinformatics algorithms
)
