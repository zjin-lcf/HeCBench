# axhelm-hip/CMakeLists.txt

# enable Fortran for BlasLapack
enable_language(Fortran)

# specify a directory to unzip the library files 
set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/BlasLapack.tar.gz")
FILE(GLOB FORTRAN_SOURCES "${SRC_DIR}/BlasLapack/*.f")
add_library(BlasLapack_hip STATIC ${FORTRAN_SOURCES})

add_hecbench_benchmark(
    NAME axhelm
    MODEL hip
    SOURCES main.cu meshBasis.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -Ddfloat=float -Ddlong=int
    LINK_LIBRARIES BlasLapack_hip
)
