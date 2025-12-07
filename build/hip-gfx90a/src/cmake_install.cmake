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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gaussian-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gaussian-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gaussian-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gaussian-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/geodesic-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/geodesic-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/geodesic-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/geodesic-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/glu-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/glu-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/glu-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/glu-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gmm-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gmm-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gmm-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/gmm-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/heartwall-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/heartwall-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/heartwall-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hmm-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hmm-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hmm-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hmm-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hotspot-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hotspot-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hotspot-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hybridsort-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/hybridsort-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/idivide-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/idivide-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/idivide-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/idivide-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/interleave-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/interleave-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/interleave-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/interleave-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/inversek2j-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/inversek2j-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/inversek2j-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/inversek2j-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ising-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ising-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ising-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/ising-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/knn-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/knn-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/knn-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/knn-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lebesgue-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lebesgue-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lebesgue-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lebesgue-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lud-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lud-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lud-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/lud-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md5hash-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md5hash-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md5hash-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/md5hash-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/minimod-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/minimod-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/minimod-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/minkowski-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/minkowski-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/minkowski-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/minkowski-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/mixbench-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/mixbench-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/mixbench-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/mixbench-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nn-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nn-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nn-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nn-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nw-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nw-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nw-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/nw-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/particlefilter-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/particlefilter-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/particlefilter-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/particlefilter-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/pathfinder-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/pathfinder-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/pathfinder-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/pathfinder-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/perplexity-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/perplexity-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/perplexity-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/perplexity-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/qtclustering-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/qtclustering-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/qtclustering-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/qtclustering-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/radixsort-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/radixsort-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/radixsort-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/radixsort-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/randomAccess-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/randomAccess-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/randomAccess-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/randomAccess-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/remap-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/remap-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/remap-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/sad-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/sad-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/sad-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/sad-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/scan-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/scan-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/scan-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/scan-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/scatter-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/scatter-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/scatter-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/srad-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/srad-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/srad-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/sssp-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/sssp-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/sssp-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/streamcluster-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/streamcluster-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/streamcluster-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/streamcluster-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/su3-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/su3-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/su3-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/su3-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/triad-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/triad-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/triad-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/triad-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/tsa-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/tsa-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/tsa-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/tsa-omp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/xsbench-cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/xsbench-hip/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/xsbench-sycl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/vjj/dev/2025-12-05/HeCBench/build/hip-gfx90a/src/xsbench-omp/cmake_install.cmake")
endif()

