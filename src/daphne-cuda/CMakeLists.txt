cmake_minimum_required(VERSION 3.18)
project(daphne LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Default to sm_60 if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

add_executable(daphne
    src/common/main.cpp
    src/points2image/kernel.cu
)

target_include_directories(daphne PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_compile_options(daphne PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
