# Install script for directory: /home/vjj/dev/2025-12-05/HeCBench/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/jacobi-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/jacobi-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/jacobi-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/jacobi-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bfs-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bfs-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bfs-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bfs-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/softmax-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/softmax-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/softmax-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/softmax-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/attention-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/attention-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/attention-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/attention-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/accuracy-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/accuracy-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/accuracy-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/accuracy-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ace-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ace-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ace-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ace-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adam-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adam-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adam-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adam-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adamw-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adamw-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adamw-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/adamw-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aes-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aes-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aes-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aes-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/affine-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/affine-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/affine-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/affine-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aidw-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aidw-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aidw-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aidw-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/amgmk-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/amgmk-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/amgmk-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/amgmk-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aobench-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aobench-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aobench-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/aobench-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/backprop-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/backprop-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/backprop-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/backprop-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bilateral-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bilateral-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bilateral-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/bilateral-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/cfd-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/cfd-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/cfd-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/cfd-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/clenergy-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/clenergy-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/clenergy-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/clenergy-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/dct8x8-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/dct8x8-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/dct8x8-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/dct8x8-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/fft-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/fft-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/fft-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/fft-omp/cmake_install.cmake")
endif()

