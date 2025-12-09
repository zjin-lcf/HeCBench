# sss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL hip
    SOURCES DPmixGGM.cpp DPmixGGM_Lists.cpp DPmixGGM_SSSmoves.cpp graph.cpp gwish.cpp kernels.cu main.cu utilities.cpp
    CATEGORIES algorithms
)
