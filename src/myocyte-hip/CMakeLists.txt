# myocyte-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME myocyte
    MODEL hip
    SOURCES embedded_fehlberg_7_8.cu kernel.cu kernel_cam.cu kernel_ecc.cu kernel_fin.cu main.cu master.cu solver.cu work.cu
    CATEGORIES algorithms
)
