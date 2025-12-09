# sss-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL sycl
    SOURCES DPmixGGM.cpp DPmixGGM_Lists.cpp DPmixGGM_SSSmoves.cpp graph.cpp gwish.cpp kernels.cpp main.cpp utilities.cpp
    CATEGORIES algorithms
)
