# oneAPI Direct Programming
This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA, SYCL, and OpenMP target offloading. Intel<sup>®</sup> DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program. 


# Experiments
We compare the performance of the SYCL, DPCT-generated, and OpenMP implementations of each program. The performance results below were obtained with the [Intel OpenCL intercept layer](https://github.com/intel/opencl-intercept-layer). "total enqueue" indicates the total number of OpenCL enqueue API calls. The host timing is the total elapsed time of executing OpenCL API functions on a host while the device timing is the total elapsed time of executing OpenCL API functions on a GPU. The Plugin Interface is OpenCL.  
 
## Setup
Software: Intel<sup>®</sup> oneAPI Beta08 Toolkit, Ubuntu 18.04  
Platform 1: Intel<sup>®</sup> Xeon E3-1284L with a Gen8 P6300 integrated GPU  
Platform 2: Intel<sup>®</sup> Xeon E-2176G with a Gen9.5 UHD630 integrated GPU

## Run
A script "run.sh" attempts to run all tests with the OpenCL plugin interface. To run a single test, go to a test directory and type the command "make run".  

## Results on Platform 1
| amgmk | SYCL | DPCT | OpenMP |     
| --- | --- | --- | --- |            
| total enqueue | 501 | 506 | 2010 | 
| host timing(s) | 0.41 | 0.88 | 3.78 | 
| device timing(s) | 0.18 | 0.18 | 0.18 |  


| aobench | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 20 | 30 | 85 |
| host timing(s) | 0.58 | 0.89 | 3.71 | 
| device timing(s) | 0.13 | 0.13 | 0.16 |  


| bezier-surface | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 12 |
| host timing(s) | 1.5 | 1.79 | 4.47 | 
| device timing(s) | 0.7 | 0.71 | 0.75 |  


| bitonic-sort | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 326 | 327 | 1957 |
| host timing(s) | 2.21 | 2.56 | 5.85 | 
| device timing(s) | 1.92 | 1.93 | 2.36 |  


| black-scholes | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 9 |
| host timing(s) | 0.57 | 1.42 | 4.67 | 
| device timing(s) | 0.16 | 0.35 | 0.95 |  


| chi2 | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 12 |
| host timing(s) | 1.1 | 1.41 | 4.5 |
| device timing(s) | 0.19 | 0.23 | 0.92 |


| gamma-correction | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 9 |
| host timing(s) | 0.27 | 0.66 | 3.56 |
| device timing(ms) | 14 | 27 | 73 |


| haccmk | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 4 | 11 | 21 |
| host timing(s) | 0.21 | 0.72 | 3.49 |
| device timing(ms) | 6.8 | 6.7 | 7.6 |


| heat | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 1003 | 1003 | 10029 |
| host timing(s) | 8.54 | 8.44 | 12.36 |
| device timing(s) | 7.98 | 7.6 | 8.36 |


| inversek2j | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 100001 | 100004 | 400008 |
| host timing(s) | 5 | 3.75 | 16.1 |
| device timing(s) | 1.93 | 2.65 | 3.85 |


| iso2dfd | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 1001 | 1004 | 10010 |
| host timing(s) | 2.18 | 2.5 | 5.87 |
| device timing(s) | 1.91 | 1.94 | 2.1 |

| mandelbrot | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 303 | 404 | 611 |
| host timing(s) | 0.22 | 0.69 | 3.49 |
| device timing(ms) | 3.29 | 3.65 | 4.25 |


| matrix-mul | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 4 | 13 |
| host timing(s) | 7 | 9.3 | 13.1 |
| device timing(s) | 6.7 | 8.68 | 9.56 |


| md | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 1002 | 1002 | 8008 |
| host timing(s) | 3.67 | 3.45 | 7.14 |
| device timing(s) | 3.3 | 2.77 | 3.36 |


| md5hash | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 16 | 28 | 57 |
| host timing(s) | 3.75 | 3.37 | 6.22 |
| device timing(s) | 2.6 | 2.6 | 2.6 |


| page-rank | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 6 | 11 | 30 |
| host timing(s) | 0.75 | 1.25 | 3.99 |
| device timing(s) | 0.23 | 0.31 | 0.36 |


| particle-diffusion | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 7 | 18 |
| host timing(s) | 1.3 | 1.7 | 4.98 |
| device timing(s) | 0.22 | 0.51 | 1.42 |


| reduction | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 325 | 326 | 975 |
| host timing(s) | 1.3 | 1.74 | 4.65 |
| device timing(s) | 1 | 0.95 | 1.13 |


| sph | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2002 | 2004 | 13512 |
| host timing(s) | 14.6 | 15.2 | 12.2 |
| device timing(s) | 14 | 14.1 | 10.9 |


| stencil | SYCL | DPCT | OpenMP |
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 10 |
| host timing(s) | 0.73 | 1.13 | 4.1 |
| device timing(s) | 0.12 | 0.19 | 0.51 |

## Results on Platform 2
| amgmk | SYCL | DPCT | OpenMP |     
| --- | --- | --- | --- |            
| total enqueue | 501 | 506 | 2010 | 
| host timing(s) | 0.59 | 1.04 | 3.87 | 
| device timing(s) | 0.28 | 0.29 | 0.28 |  


| aobench | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 20 | 30 | 85 |
| host timing(s) | 0.7 | 1.04 | 3.92 | 
| device timing(s) | 0.27 | 0.27 | 0.31 |  


| bezier-surface | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 12 |
| host timing(s) | 1.94 | 2.1 | 6.04 | 
| device timing(s) | 1.19 | 1.17 | 2.27 |  


| bitonic-sort | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 326 | 327 | 1957 |
| host timing(s) | 3.01 | 3.22 | 6.26 | 
| device timing(s) | 2.59 | 2.52 | 2.77 |  


| black-scholes | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 9 |
| host timing(s) | 0.71 | 1.42 | 4.57 | 
| device timing(s) | 0.27 | 0.42 | 1.01 |  


| chi2 | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 12 |
| host timing(s) | 0.96 | 1.25 | 4.51 |
| device timing(s) | 0.19 | 0.31 | 1.03 |


| gamma-correction | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 9 |
| host timing(s) | 0.33 | 0.65 | 3.41 |
| device timing(ms) | 16 | 26 | 68 |


| haccmk | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 4 | 11 | 21 |
| host timing(s) | 0.31 | 0.63 | 3.49 |
| device timing(ms) | 5.8 | 5.8 | 6.8 |


| heat | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 1003 | 1003 | 10029 |
| host timing(s) | 9.79 | 9.75 | 16.1 |
| device timing(s) | 9.19 | 8.74 | 12.3 |


| inversek2j | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 100001 | 100004 | 400008 |
| host timing(s) | 6.45 | 22 | 1211 |
| device timing(s) | 3.91 | 4.37 | 1181 |


| iso2dfd | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 1001 | 1004 | 10010 |
| host timing(s) | 2.77 | 3.29 | 6.42 |
| device timing(s) | 2.42 | 2.45 | 2.75 |

| mandelbrot | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 303 | 404 | 611 |
| host timing(s) | 0.34 | 0.72 | 3.48 |
| device timing(ms) | 5.18 | 5.48 | 5.15 |


| matrix-mul | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 4 | 13 |
| host timing(s) | 6.8 | 7.76 | 13.37 |
| device timing(s) | 6.44 | 7.09 | 9.95 |


| md | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 1002 | 1002 | 8008 |
| host timing(s) | 7.52 | 3.2 | 10.85 |
| device timing(s) | 7.17 | 2.42 | 7.17 |


| md5hash | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 16 | 28 | 57 |
| host timing(s) | 6.56 | 6.13 | 8.65 |
| device timing(s) | 5.39 | 5.39 | 5.12 |


| page-rank | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 6 | 11 | 30 |
| host timing(s) | 1.22 | 1.67 | 4.3 |
| device timing(s) | 0.71 | 0.77 | 0.82 |


| particle-diffusion | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 7 | 18 |
| host timing(s) | 1.14 | 1.49 | 5.05 |
| device timing(s) | 0.24 | 0.48 | 1.52 |


| reduction | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 325 | 326 | 975 |
| host timing(s) | 1.61 | 1.93 | 4.8 |
| device timing(s) | 1.2 | 1.17 | 1.3 |


| sph | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2002 | 2004 | 13512 |
| host timing(s) | 21.8 | 22.3 | 24.8 |
| device timing(s) | 21.1 | 21 | 20.4 |


| stencil | SYCL | DPCT | OpenMP |
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 10 |
| host timing(s) | 0.68 | 0.89 | 4 |
| device timing(s) | 0.09 | 0.14 | 0.48 |

# Reference
### amgmk 
  The relax kernel in the AMGmk benchmark (https://asc.llnl.gov/CORAL-benchmarks/Micro/amgmk-v1.0.tar.gz)
  
### aobench
  A lightweight ambient occlusion renderer (https://code.google.com/archive/p/aobench)

### bezier-surface
  The Bezier surface (https://github.com/chai-benchmarks/chai)

### bitonic-sort
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### black-scholes
  The Black Scholes simulation (https://github.com/cavazos-lab/FinanceBench)

### chi2
  The Chi-square 2-df test. The input data must be downloaded before running the test. Please see README for the link.

### gamma-correction
  Gamma correction (https://github.com/intel/BaseKit-code-samples)

### haccmk
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

## heat
  The heat equation solver (https://github.com/UoB-HPC/heat_sycl)

### inversek2j
  The inverse kinematics for 2-joint arm (http://axbench.org/)

### iso2dfd, mandelbrot, particle-diffusion
  The HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### matrix-mul
  Single-precision floating-point matrix multiply

### md
  Molecular dynamics function in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### md5hash
  The MD5 Hash function (https://github.com/vetter/shoc/)

### page-rank
  PageRank (https://github.com/Sable/Ostrich/tree/master/map-reduce/page-rank)

### reduction
  Integer sum reduction (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/reduction)

### xsbench
  The nuclear reactor simulation proxy applications in SYCL  
  (https://github.com/ANL-CESAR/XSBench/tree/master/sycl)

### sph
  The simple n^2 SPH simulation (https://github.com/olcf/SPH_Simple)

### stencil
  1D stencil using shared memory


## Development Team
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
Results presented were obtained using the Chameleon testbed supported by the National Science Foundation and the Intel<sup>®</sup> DevCloud.
