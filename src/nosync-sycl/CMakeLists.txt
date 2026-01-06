# nosync-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nosync
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
