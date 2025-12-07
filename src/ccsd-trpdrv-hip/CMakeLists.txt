# ccsd-trpdrv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccsd-trpdrv
    MODEL hip
    SOURCES ccsd_tengy.cu
    CATEGORIES algorithms
)
