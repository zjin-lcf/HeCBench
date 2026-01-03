# logan-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME logan
    MODEL hip
    SOURCES src/main.cu src/seed.cu src/score.cu src/logan_functions.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    COMPILE_OPTIONS -fgpu-rdc -DADAPTABLE
    LINK_LIBRARIES -fopenmp -fgpu-rdc --hip-link
    CATEGORIES bioinformatics
)
