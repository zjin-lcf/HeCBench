# black-scholes-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME black-scholes
    MODEL omp
    SOURCES blackScholesAnalyticEngine.cpp blackScholesAnalyticEngineKernels.cpp blackScholesAnalyticEngineKernelsCpu.cpp
    CATEGORIES algorithms
)
