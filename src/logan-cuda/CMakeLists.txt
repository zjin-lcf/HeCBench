# logan-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME logan
    MODEL cuda
    SOURCES
        src/main.cu
        src/seed.cu
        src/score.cu
        src/logan_functions.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    COMPILE_OPTIONS -Xcompiler=-fopenmp -maxrregcount=32 -DADAPTABLE
    LINK_LIBRARIES -lgomp
    CATEGORIES bioinformatics algorithms
)
