# xlqc-cuda/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/example.tar.gz")

add_hecbench_benchmark(
    NAME xlqc
    MODEL cuda
    SOURCES cuda_rys_dp.cu cuda_rys_sp.cu main.cu
            scf.cc
            basis.cc
            int_lib/crys.cc
            int_lib/cints.cc
    INCLUDE_DIRS ${SRC_DIR}/int_lib
    CATEGORIES algorithms
    COMPILE_OPTIONS -dc
    LINK_LIBRARIES gsl gslcblas
)
