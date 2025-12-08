# minimap2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME minimap2
    MODEL sycl
    SOURCES
        src/host_data_io.cpp
        src/host_kernel.cpp
        src/main.cpp
        src/memory_scheduler.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    CATEGORIES bioinformatics algorithms
)
