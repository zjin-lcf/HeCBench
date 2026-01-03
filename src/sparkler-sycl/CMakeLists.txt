# sparkler-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sparkler
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES mpi -qmkl=parallel
)
