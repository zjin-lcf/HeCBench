# black-scholes-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME black-scholes
    MODEL sycl
    SOURCES blackScholesAnalyticEngine.cpp blackScholesAnalyticEngineKernels.cpp blackScholesAnalyticEngineKernelsCpu.cpp
    CATEGORIES algorithms
)
