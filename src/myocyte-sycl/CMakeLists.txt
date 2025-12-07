# myocyte-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME myocyte
    MODEL sycl
    SOURCES embedded_fehlberg_7_8.cpp kernel.cpp kernel_cam.cpp kernel_ecc.cpp kernel_fin.cpp main.cpp master.cpp solver.cpp work.cpp
    CATEGORIES algorithms
)
