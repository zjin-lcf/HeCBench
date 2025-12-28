# local-ht-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME local-ht
    MODEL hip
    SOURCES
        src/main.cpp
        src/kernel.cpp
        src/helper.cpp
    CATEGORIES bioinformatics
)
