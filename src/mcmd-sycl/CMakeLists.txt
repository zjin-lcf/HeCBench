# mcmd-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mcmd-cuda")

add_hecbench_benchmark(
    NAME mcmd
    MODEL sycl 
    SOURCES force_kernel.cpp ${INC_DIR}/main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES algorithms
    COMPILE_OPTIONS -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -DSYCL
)
