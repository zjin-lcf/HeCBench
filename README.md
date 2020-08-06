# oneAPI Benchmarks
This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA, SYCL, and OpenMP target offloading. Intel DPC++ Compatibility Tool can convert a CUDA program to a SYCL program. 


# Experiments
We compare the performance of the SYCL, DPCT-generated, and OpenMP implementations of each program. The performance results were obtained with the [Intel OpenCL intercept layer](https://github.com/intel/opencl-intercept-layer). "total enqueue" indicates the total number of OpenCL enqueue API calls. The host timing is the total elapsed time of executing OpenCL API functions on a host while the device timing is the total elapsed time of executing OpenCL API functions on a GPU.
 
## Setup
Software: Intel oneAPI Beta08 Toolkit, Ubuntu 18.04  
Platform: Intel Xeon E3-1284L with a Gen8 P6300 integrated GPU

## Results
| amgmk | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 501 | 506 | 2010 |
| host timing(s) | 0.41 | 0.88 | 3.78 | 
| device timing(s) | 0.18 | 0.18 | 0.18 |  


| aobench | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 20 | 30 | 85 |
| host timing(s) | 0.58 | 0.89 | 3.92 | 
| device timing(s) | 0.13 | 0.13 | 0.16 |  


| bezier-surface | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 12 |
| host timing(s) | 1.5 | 1.79 | 5.3 | 
| device timing(s) | 0.7 | 0.71 | 1.3 |  


| bitonic-sort | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 326 | 327 | 1957 |
| host timing(s) | 2.21 | 2.56 | 5.85 | 
| device timing(s) | 1.92 | 1.93 | 2.36 |  


| chi2 | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 12 |
| host timing(s) | 1.1 | 1.41 | 4.5 |
| device timing(s) | 0.19 | 0.23 | 0.92 |


| gamma-correction | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 9 |
| host timing(s) | 0.27 | 0.66 | 3.6 |
| device timing(ms) | 14 | 27 | 73 |


| haccmk | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 4 | 11 | 21 |
| host timing(s) | 0.21 | 0.72 | 3.49 |
| device timing(ms) | 6.8 | 6.7 | 7.6 |


| inversek2j | SYCL | DPCT | OpenMP | 
| --- | --- | --- | --- |
| total enqueue | 100001 | 100004 | 400008 |
| host timing(s) | 5 | 3.75 | 41.7 |
| device timing(s) | 1.93 | 2.65 | 28.7 |


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
| host timing(s) | 14.6 | 15.2 | 17.4 |
| device timing(s) | 14 | 14.1 | 13.3 |


| stencil | SYCL | DPCT | OpenMP |
| --- | --- | --- | --- |
| total enqueue | 2 | 3 | 10 |
| host timing(s) | 0.73 | 1.13 | 4.1 |
| device timing(s) | 0.12 | 0.19 | 0.51 |


# Reference
### amgmk 
  The relax kernel in the AMGmk benchmark (https://asc.llnl.gov/CORAL-benchmarks/Micro/amgmk-v1.0.tar.gz)
  
### aobench
  A lightweight ambient occlusion renderer (https://code.google.com/archive/p/aobench)

### bezier-surface
  The Bezier surface (https://github.com/chai-benchmarks/chai)

### bitonic-sort
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### chi2
  The Chi-square 2-df test 

### gamma-correction
  Gamma correction (https://github.com/intel/BaseKit-code-samples)

### haccmk
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

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
