# oneAPI Direct Programming
This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA, SYCL, and OpenMP target offloading. Intel<sup>®</sup> DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program in which memory management migration is implemented using the explicit and restricted Unified Shared Memory extension (DPCT usm) or the DPCT header files (DPCT header) .


# Experiments
We compare the performance of the SYCL, DPCT-generated, and OpenMP implementations of each program. The performance results below were obtained with the [Intel OpenCL intercept layer](https://github.com/intel/opencl-intercept-layer). "total enqueue" indicates the total number of low-level OpenCL enqueue commands called by a parallel program. These enqueue commands include "clEnqueueNDRangeKernel", "clEnqueueReadBuffer", and "clEnqueueWriteBuffer". The host timing is the total elapsed time of executing OpenCL API functions on a CPU host while the device timing is the total elapsed time of executing OpenCL API functions on a GPU device. The Plugin Interface is OpenCL.  
 
## Setup
Software: Intel<sup>®</sup> oneAPI Beta08 Toolkit, Ubuntu 18.04  
Platform 1: Intel<sup>®</sup> Xeon E3-1284L with a Gen8 P6300 integrated GPU  
Platform 2: Intel<sup>®</sup> Xeon E-2176G with a Gen9.5 UHD630 integrated GPU

## Run
A script "run.sh" attempts to run all tests with the OpenCL plugin interface. To run a single test, go to a test directory and type the command "make run".  

## Results on Platform 1
| amgmk | SYCL | DPCT usm | DPCT header | OpenMP |     
| --- | --- | --- | --- | --- |            
| total enqueue | 501 | 506 | 506 | 2010 | 
| host timing(s) | 0.41 | 0.88 | 0.88 | 3.78 | 
| device timing(s) | 0.18 | 0.18 | 0.18 | 0.18 |  


| aobench | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 20 | 30 | 30 | 85 |
| host timing(s) | 0.58 | 0.92 | 0.95 | 3.71 | 
| device timing(s) | 0.14 | 0.14 | 0.14 | 0.16 |  


| atomicIntrinsics | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 3 | 3 | 3 | NA |
| host timing(s) | 9.3 | 9.7 | 9.7 | NA | 
| device timing(s) | 9.1 | 9.1 | 9.1 | NA |  


| backprop | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 5 | 9 | 9 | 19 |
| host timing(s) | 2.0 | 2.4 | 2.8 | 6.1 | 
| device timing(s) | 0.77 | 1.49 | 1.49 | 2.3 |  


| bezier-surface | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 12 |
| host timing(s) | 1.5 | 1.79 | 1.87 | 4.47 | 
| device timing(s) | 0.7 | 0.71 | 0.72 | 0.75 |  


| bitonic-sort | SYCL | DPCT usm | DPCT header| OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 326 | 327 | 327 | 1957 |
| host timing(s) | 2.21 | 2.56 | 2.67 | 5.85 | 
| device timing(s) | 1.92 | 1.93 | 1.97 | 2.36 |  


| black-scholes | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 9 |
| host timing(s) | 0.57 | 1.42 | 1.46 | 4.67 | 
| device timing(s) | 0.16 | 0.35 | 0.34 | 0.95 |  


| bspline-vgh | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 120003 |  120007 |  120007 | 228009 | 
| host timing(s) | 5.8 | 7.6 | 8.1 | 8.4 |
| device timing(s) | 0.67 | 2.38 | 2.15 | 1.36 |


| b+tree | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 5 | 20 | 20 | 32 |
| host timing(s) | 3.5 | 0.58 | 0.65 | 3.48 |
| device timing(s) | 3.1 | 0.0068 | 0.0068 | 0.0082 |


| cfd | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 16005 | NA | NA | 132035 |
| host timing(s) | 4.4 | NA | NA | 9.95 |
| device timing(s) | 3.5 | NA | NA | 3.76 |


| chi2 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 12 |
| host timing(s) | 1.1 | 1.41 | 1.47 | 4.5 |
| device timing(s) | 0.19 | 0.23 | 0.35 | 0.92 |


| clenergy | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 502 | 502  | 502 | 3011 |
| host timing(s) | 12.3 | 11.6 | 11.8 | 14.9 |
| device timing(s) | 11.8 | 10.8 | 10.9 | 11.2  |


| clink | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 40 | 160  | 160 | 205 |
| host timing(s) | 19.5 | 20.4 | 24.3 | 25.1 |
| device timing(s) | 13.5 | 13.3 | 16.9 | 17.6  |


| cobahh | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 17 | 24 | 24 | 109 |
| host timing(s) | 2.83 | 4.2 | 4.4 | 7.6 |
| device timing(s) | 2.53 | 3.2 | 3.2 | 4.0 |


| compute-score | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 201 | 207  | 207 | 812 |
| host timing(s) | 8.4 | 8.3 | 9.1 | 20.2 |
| device timing(s) | 8.0 | 7.4 | 8.1 | 16.3  |


| filter | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 3 | 5 | 5 | 11 |
| host timing(s) | 0.62 | 0.92 | 0.98 | 6.4 |
| device timing(ms) | 85 | 147 | 142 |  2711 |


| fft | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 201 | 202 | 202 | NA |
| host timing(s) | 16.4 | 19.5 | 19.5 | NA |
| device timing(ms) | 14.3 | 17.1 | 17.0 | NA | 


| gamma-correction | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 9 |
| host timing(s) | 0.27 | 0.66 | 0.69 | 3.56 |
| device timing(ms) | 14 | 27 | 24 |  73 |


| gaussian | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 8193 | 8196 | 8196 | 61437 |
| host timing(s) | 11.6 | 11.8 | 12.7 | 14.7 |
| device timing(s) | 11.0 | 11.1 | 11.8 | 9.6 |


| geodesic | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 | 407 |
| host timing(s) | 5.5 | 5.8 | 5.8 | 9.2 |
| device timing(s) | 5.1 | 5.1 | 5.1 | 5.6 |


| haccmk | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 4 | 11 | 11 | 21 |
| host timing(s) | 0.21 | 0.72 | 0.6 | 3.49 |
| device timing(ms) | 6.8 | 6.7 | 6.8 | 7.6 |


| heartwall | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 212 | 220 | 220 | 637 |
| host timing(s) | 17.2 | 9.1 | 9.5 |  11.6 |
| device timing(s) | 16.1 | 8.3 | 8.4 | 7.92 |


| heat | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1003 | 1003 | 1003 | 10029 |
| host timing(s) | 8.54 | 8.44 | 8.94 | 12.36 |
| device timing(s) | 7.98 | 7.6 | 7.92 | 8.36 |


| heat2d | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 102 | 102 | 1107 |
| host timing(s) | 1.99 | 2.33 | 2.3 | 5.5 |
| device timing(s) | 1.65 | 1.68 | 1.63 | 1.89 |


| histogram | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1218 | 1221 | 1221 | NA |
| host timing(s) | 1.88 | 1.39 | 1.59 | NA |
| device timing(s) | 0.57 | 0.56 | 0.59 | NA |


| hmm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 501 | 504 | 504 | 3249 |
| host timing(s) | 8.9 | 11.4 | 11.2 | 14.1 |
| device timing(s) | 8.6 | 10.7 | 10.5 | 10.4 |


| hotspot3D | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 5001 | 5003 | 5003 | 90008 |
| host timing(s) | 4.5 | 4.6 | 4.9 | 9.4 |
| device timing(s) | 4.1 | 4.1 | 4.1 | 4.2 |


| hybridsort | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 25 | 33 | 33 | 193 |
| host timing(s) | 1.5 | 1.74 | 1.87 | 4.89 |
| device timing(s) | 0.82 | 0.87 | 0.86 | 1.21 |


| inversek2j | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100001 | 100004 | 100004 | 400008 |
| host timing(s) | 5 | 3.75 | 5.5 | 16.1 |
| device timing(s) | 1.93 | 2.65 | 1.99 | 3.85 |


| ising | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 404 | 405 | 405 | 4018 |
| host timing(s) | 6.2 | 4.4 | 4.3 |  9.9 |
| device timing(s) | 5.8 | 3.67 | 3.49 | 6.2 |


| iso2dfd | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1001 | 1004 | 1004 | 10010 |
| host timing(s) | 2.18 | 2.5 | 2.6 |  5.87 |
| device timing(s) | 1.91 | 1.94 | 1.92 | 2.1 |


| keccaktreehash | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 24 | 27 | 27 | 62 |
| host timing(s) | 0.95 | 1.33 | 1.36 |  17.8 |
| device timing(s) | 0.57 | 0.58 | 0.57 | 14.1 |


| kmeans | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 21500 | 21501 | 21501 | 71703 |
| host timing(s) | 110 | 112 | 114 |  116 |
| device timing(s) | 106 | 109 | 110 | 111 |


| knn | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 500 | 700 | 700 | 2007 |
| host timing(s) | 9.8 | 12.3 | 12.5 |  16.2 |
| device timing(s) | 7.6 | 10.1 | 10.4 | 10.7 |


| laplace | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 16 | 28 | 28 | 57 |
| host timing(s) | 3.75 | 3.37 | 3.59 | 6.22 |
| device timing(s) | 2.6 | 2.6 | 2.8 | 2.6 |


| lavaMD | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 6 | 6 | 15 |
| host timing(s) | 2.8 | 1.4 | 1.48 | 4.4 |
| device timing(s) | 2.5 | 0.77 | 0.76 | 0.8 |


| leukocyte | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 204 | 711 | 711 | 1334 |
| host timing(s) | 2.3 | 2.97 | 3.0 | 5.8 |
| device timing(s) | 1.99 | 2.17 | 2.17 | 2.14 |


| lud | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1535 | 1536 | 1536 | 6145 |
| host timing(s) | 8.9 | 11.0 | 11.2 | 14.1 |
| device timing(s) | 7.8 | 9.7 | 9.8 | 9.7 |


| memcpy | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 131072 | 131072 | 131072 | 131072 |
| host timing(s) | 4.3 | 4.9 | 4.6 | 2.4 |
| device timing(s) | 1.2 | 1.5 | 1.2 | 1.7 |


| miniFE | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2407 | NA | NA | 6645 |
| host timing(s) | 9.9 | NA | NA | NA | 16.7 |
| device timing(s) | 8.95 | NA | NA | 12.9 | 


| mixbench | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2049 |  2050 |  2050 | 6151 | 
| host timing(s) | 5.1 | 5.5 | 5.6 | 9.6 |
| device timing(s) | 4.8 | 4.8 | 4.8 | 5.8 |


| mkl-sgemm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 80001 | 120004 | 80004 | 80004 |
| host timing(s) | 6.3 | 8.3 | 6.6 | 4.7 |
| device timing(s) | 2.38 | 2.52 | 2.38 | 2.53 |


| nbody | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 402 | 402 | 402 | 1308 |
| host timing(s) | 2.7 | 3.0 | 3.1 | 6.2 |
| device timing(s) | 2.4 | 2.4 | 2.4 | 2.7 |


| nn | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 13 |
| host timing(s) | 0.3 | 0.60 | 0.65 | 3.1 |
| device timing(us) | 37 | 57 | 62 | 220 |


| nw | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2048 | 2050 | 2050 | 13314 |
| host timing(s) | 2.0 | 2.4 | 2.4 | 5.6 |
| device timing(s) | 0.51 | 0.85 | 0.79 | 1.51 |


| page-rank | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 6 | 11 | 11 | 30 |
| host timing(s) | 0.75 | 1.25 | 1.19 | 3.99 |
| device timing(s) | 0.23 | 0.31 | 0.31 | 0.36 |


| particle-diffusion | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 7 | 7 | 18 | 
| host timing(s) | 1.3 | 1.7 | 1.8 | 4.98 |
| device timing(s) | 0.22 | 0.51 | 0.53 | 1.42 |


| pathfinder | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 104 | 104 | 709 | 
| host timing(s) | 2.36 | 5.72 | 5.65 | 11.8 |
| device timing(s) | 1.99 | 5.0 | 4.98 | 8.1 |


| projectile | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 102 | 102 | 307 | 
| host timing(s) | 3.0 | 3.4 | 3.5 | 6.5 |
| device timing(s) | 2.7 | 2.7 | 2.7 | 2.85 |


| randomAccess | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 4 | 4 | 4 | 21 |
| host timing(s) | 2.6 | 2.9 | 2.9 | 7.1 | 
| device timing(s) | 2 | 2.1 | 2.0 | 3.3 |


| reduction | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 325 | 326 | 326 | 975 |
| host timing(s) | 1.3 | 1.74 | 1.8 | 4.65 |
| device timing(s) | 1 | 0.95 | 1.0 | 1.13 |


| reverse | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1048578 | 1048579 | 1048579 | 1048584 |
| host timing(s) | 33.4 | 27.1 | 74.9 | 43.3 |
| device timing(s) | 2.3 | 2.44 | 1.47 | 4.14 |


| rng-wallace | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 200 | 202 | 202 |  NA |
| host timing(s) | 4.3 | 3.9 | 4.6 | NA | 
| device timing(s) | 3.6 | 3.2 | 3.6 | NA |


| rsbench | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 10 | 10 | 68 |
| host timing(s) | 11.6 | 11.8 | 11.8 | 20.8 |
| device timing(s) | 8.8 | 8.8 | 8.6 | 12.1 |


| rtm8 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 32 | 38 | 38 | 220 |
| host timing(s) | 4.7 | 4.8 | 5.0 | 8.7 |
| device timing(s) | 3.9 | 3.9 | 3.95 | 4.9 |


| randomAccess | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 4 | 4 | 4 | 21 |
| host timing(s) | 2.6 | 2.9 | 2.9 | 7.1 | 
| device timing(s) | 2 | 2.1 | 2.0 | 3.3 |


| reduction | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 325 | 326 | 326 | 975 |
| host timing(s) | 1.3 | 1.74 | 1.8 | 4.65 |
| device timing(s) | 1 | 0.95 | 1.0 | 1.13 |


| reverse | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1048578 | 1048579 | 1048579 | 1048584 |
| host timing(s) | 33.4 | 27.1 | 74.9 | 43.3 |
| device timing(s) | 2.3 | 2.44 | 1.47 | 4.14 |


| rng-wallace | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 200 | 202 | 202 |  NA |
| host timing(s) | 4.3 | 3.9 | 4.6 | NA | 
| device timing(s) | 3.6 | 3.2 | 3.6 | NA |


| rsbench | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 10 | 10 | 68 |
| host timing(s) | 11.6 | 11.8 | 11.8 | 20.8 |
| device timing(s) | 8.8 | 8.8 | 8.6 | 12.1 |


| rtm8 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 32 | 38 | 38 | 220 |
| host timing(s) | 4.7 | 4.8 | 5.0 | 8.7 |
| device timing(s) | 3.9 | 3.9 | 3.95 | 4.9 |


| scan | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100001 | 10002 | 10002 | 200007 |
| host timing(s) | 3.4 | 2.9 | 3.8 | 10.3 |
| device timing(s) | 0.69 | 1.22 | 0.9 | 1.39 |


| softmax | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 |  507 |
| host timing(s) | 1.6 | 4.5 | 1.9 | 5.1 |
| device timing(s) | 1.3 | 3.8 | 1.3 | 1.5 |


| sph | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2002 | 2004 | 2004 |  13512 |
| host timing(s) | 14.6 | 15.2 | 15.4 | 12.2 |
| device timing(s) | 14 | 14.1 | 14.2 | 10.9 |


| srad | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 8003 | 8008 | 8008 | 36026 |
| host timing(s) | 1.3 | 1.54 | 1.79 | 5.0 |
| device timing(s) | 0.62 | 0.78 | 0.77 | 0.84 |


| sssp | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 88395 | 88626 | 88655 | NA |
| host timing(s) | 6.6 | 8.1 | 8.4 | NA |
| device timing(s) | 2.3 | 2.2 | 2.2 | NA |


| stencil | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 10 | 
| host timing(s) | 0.73 | 1.13 | 1.12 | 4.1 |
| device timing(s) | 0.12 | 0.19 | 0.18 | 0.51 |


| su3 | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 104 | 104 | 715 |
| host timing(s) | 7.9 | 8.2 | 8.3 | 11 |
| device timing(s) | 7.4 | 7.4 | 7.4 | 7.4 |


| xsbench | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 9 | 9 | 26 |
| host timing(s) | 3.0 | 3.4 | 3.0 | 6.8 |
| device timing(s) | 2.6 | 2.7 | 2.1 | 3.1 |

## Results on Platform 2
| all-pairs-distance | SYCL | DPCT usm | DPCT header | OpenMP |     
| --- | --- | --- | --- | --- |            
| total enqueue | 60 | 61 | 61 | 67 | 
| host timing(s) | 0.49 | 0.87 | 0.93 | 58 | 
| device timing(s) | 0.14 | 0.22 | 0.22 | 54 |  


| amgmk | SYCL | DPCT usm | DPCT header | OpenMP |     
| --- | --- | --- | --- | --- |           
| total enqueue | 501 | 506 | 506 | 2010 | 
| host timing(s) | 0.59 | 1.04 | 0.95 | 3.87 | 
| device timing(s) | 0.28 | 0.29 | 0.28 | 0.28 |  


| aobench | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | ---- |
| total enqueue | 20 | 30 | 30 | 85 |
| host timing(s) | 0.7 | 1.04 | 1.02 | 3.92 | 
| device timing(s) | 0.27 | 0.27 | 0.27 | 0.31 |  


| atomicIntrinsics | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 3 | 3 | 3 | NA |
| host timing(s) | 1.0 | 1.37 | 1.35 | NA | 
| device timing(s) | 0.73 | 0.73 | 0.73 | NA |  


| backprop | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 5 | 9 | 9 | 19 |
| host timing(s) | 1.6 | 1.98 | 2.66 | 5.8 | 
| device timing(s) | 0.66 | 1.16 | 1.15 | 1.9 |  


| bezier-surface | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 12 | 
| host timing(s) | 1.94 | 2.1 | 2.2 | 5.37 | 
| device timing(s) | 1.19 | 1.17 | 1.18 | 0.81 |  


| bfs | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 52 | 55 | 55 | 133 |
| host timing(s) | 0.4 | 0.7 | 0.73 | 3.54 | 
| device timing(s) | 0.23 | 0.27 | 0.26 | 0.36 |  


| bitonic-sort | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 326 | 327 | 327 | 1957 |
| host timing(s) | 3.01 | 3.22 | 3.21 | 6.26 | 
| device timing(s) | 2.59 | 2.52 | 2.52 | 2.77 |  


| black-scholes | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 9 |
| host timing(s) | 0.71 | 1.42 | 1.43 | 4.49 | 
| device timing(s) | 0.27 | 0.42 | 0.37 | 0.96 |  


| bspline-vgh | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 120003 | 120007 | 120007 | 228009 | 
| host timing(s) | 5.3 | 9.5 | 6.6 | 10.4 |
| device timing(s) | 0.68 | 1.41 | 1.49 | 1.29 |


| b+tree | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 5 | 20 | 20 | 32 |
| host timing(s) | 1.04 | 0.68 | 0.69 | 3.44 |
| device timing(s) | 0.56 | 0.0073 | 0.0065 | 0.0075 |


| cfd | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 16005 | NA | NA | 132035 |
| host timing(s) | 4.4 | NA | NA | 9.95 |
| device timing(s) | 3.5 | NA | NA | 3.76 |


| chi2 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 12 |
| host timing(s) | 0.96 | 1.25 | 1.3 | 4.51 |
| device timing(s) | 0.19 | 0.31 | 0.28 | 1.03 |


| clenergy | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 502 | 502  | 502 | 3011 |
| host timing(s) | 16.7 | 16.4 | 16.8 | 20.1 |
| device timing(s) | 15.99 | 15.7 | 15.93 | 15.99  |


| clink | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 40 | 160  | 160 | 205 |
| host timing(s) | 22.6 | 24.2 | 25.4 | 30.4 |
| device timing(s) | 19.2 | 19.8 | 20.7 | 24.7  |


| cobahh | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 17 | 24 | 24 | 109 |
| host timing(s) | 1.93 | 3.0 | 3.1 | 6.56 |
| device timing(s) | 1.56 | 2.0 | 1.99 | 3.0 |


| compute-score | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 201 | 207  | 207 | 812 |
| host timing(s) | 6.9 | 7.1 | 7.3 | 18.4 |
| device timing(s) | 6.5 | 6.3 | 6.5 | 14.6  |


| filter | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 3 | 5 | 5 | 11 |
| host timing(s) | 0.62 | 0.87 | 0.90 | 8.4 |
| device timing(ms) | 61 | 104 | 95 |  4869 |


| fft | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 201 | 202 | 202 | NA |
| host timing(s) | 12.4 | 27.1 | 27.1 | NA |
| device timing(ms) | 11.4 | 24.9 | 24.9 | NA | 


| gamma-correction | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 9 |
| host timing(s) | 0.33 | 0.65 | 0.70 | 3.41 |
| device timing(ms) | 16 | 26 | 23 | 68 |


| gaussian | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 8193 | 8196 | 8196 | 61437 |
| host timing(s) | 11.2 | 12.0 | 11.5 | 15.2 |
| device timing(s) | 10.7 | 10.7 | 10.7 | 9.0 |


| geodesic | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 | 407 |
| host timing(s) | 10.7 | 10.8 | 10.7 | 13.8 |
| device timing(s) | 10.2 | 9.99 | 9.99 | 10.2 |


| haccmk | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 4 | 11 | 11 | 21 | 
| host timing(s) | 0.31 | 0.63 | 0.63 | 3.42 |
| device timing(ms) | 5.8 | 5.8 | 5.8 |  6.8 |


| heartwall | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 212 | 220 | 220 | 637 |
| host timing(s) | 26.6 | 9.4 | 9.7 |  14.2 |
| device timing(s) | 25.5 | 8.6 | 8.7 | 10.5 |


| heat | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1003 | 1003 | 1003 | 10029 |
| host timing(s) | 9.79 | 9.75 | 10.0 | 16.1 |
| device timing(s) | 9.19 | 8.74 | 9.11 | 12.3 |


| heat2d | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 102 | 102 | 1107 |
| host timing(s) | 2.65 | 2.98 | 2.99 | 5.26 |
| device timing(s) | 2.29 | 2.29 | 2.29 | 1.82 |


| histogram | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1218 | 1221 | 1221 | NA |
| host timing(s) | 2.75 | 2.5 | 2.6 | NA |
| device timing(s) | 0.83 | 0.86 | 0.85 | NA |


| hmm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 501 | 504 | 504 | 6499 |
| host timing(s) | 11.3 | 11.8 | 11.7 | 25.7 |
| device timing(s) | 10.9 | 11.0 | 10.9 | 21.9 |


| hotspot3D | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 5001 | 5003 | 5003 | 90008 |
| host timing(s) | 4.2 | 5.6 | 4.6 | 10.8 |
| device timing(s) | 3.7 | 4.0 | 3.7 | 4.1 |


| hybridsort | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 25 | 33 | 33 | 193 |
| host timing(s) | 1.5 | 1.74 | 1.87 | 4.89 |
| device timing(s) | 0.82 | 0.87 | 0.86 | 1.21 |


| inversek2j | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100001 | 100004 | 100003 | 400008 |
| host timing(s) | 6.45 | 22 | 6.9 | 50.5 |
| device timing(s) | 3.91 | 4.37 | 3.94 | 6.1  |


| ising | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 404 | 405 | 405 | 4018 |
| host timing(s) | 9.2 | 5.4 | 5.4 |  12.2 |
| device timing(s) | 8.8 | 4.6 | 4.6 | 8.7 |


| iso2dfd | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1001 | 1004 | 1004 |  10010 |
| host timing(s) | 2.77 | 3.29 | 3.15 | 6.42 |
| device timing(s) | 2.42 | 2.45 | 2.45 | 2.75 |


| keccaktreehash | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 24 | 27 | 27 | 62 |
| host timing(s) | 1.36 | 1.67 | 1.78 |  17.2 |
| device timing(s) | 0.92 | 0.96 | 0.93 | 13.7 |


| kmeans | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 21500 | 21501 | 21501 | 71703 |
| host timing(s) | 119 | 121 | 123 |  122.5 |
| device timing(s) | 114.1 | 114.1 | 114.3 | 114.6 |


| knn | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 500 | 700 | 700 | 2007 |
| host timing(s) | 12.9 | 15.3  | 14.6 | 17.0 |
| device timing(s) | 10.4 | 11.4 | 11.4 | 11.6 |


| laplace | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 108546 | 108555 | 108555  | NA |
| host timing(s) | 153 | 116 | 153 | NA |
| device timing(s) | 65 | 63 | 65 | NA |


| lavaMD | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 6 | 6 | 15 |
| host timing(s) | 1.8 | 2.0 | 2.0 | 4.8 |
| device timing(s) | 1.42 | 1.31 | 1.27 | 1.32 |


| leukocyte | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 6 | 6 | 15 |
| host timing(s) | 1.8 | 2.0 | 2.0 | 4.8 |
| device timing(s) | 1.42 | 1.31 | 1.27 | 1.32 |


| lud | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1535 | 1536 | 1536 | 6145 |
| host timing(s) | 11.1 | 11.9 | 12.0 | 15.6 |
| device timing(s) | 10.3 | 10.7 | 10.8 | 11.3 |


| mandelbrot | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 303 | 404 | 404 | 611 |
| host timing(s) | 0.34 | 0.72 | 0.74 | 3.48 |
| device timing(ms) | 5.18 | 5.48 | 5.4 | 5.15 |


| matrix-mul | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 4 | 4 | 13 |
| host timing(s) | 6.8 | 7.76 | 7.79 | 13.37 |
| device timing(s) | 6.44 | 7.09 | 7.1 | 9.95 |


| matrix-rotate | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 | 705 | 
| host timing(s) | 8.7 | 3.4 | 9.13 | 16.9 |
| device timing(s) | 8.39 | 8.42 | 8.43 | 13.0 |


| maxpool3d | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 | 1807 | 
| host timing(s) | 7.0 | 7.4 | 7.3 | 11.8 |
| device timing(s) | 6.3 | 6.4 | 6.4 | 8.2 |


| md | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1002 | 1002 | 1002 | 8008 |
| host timing(s) | 7.52 | 3.2 | 3.1 | 10.85 |
| device timing(s) | 7.17 | 2.42 | 2.41 | 7.17 |


| md5hash | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 16 | 28 | 28 | 57 | 
| host timing(s) | 6.56 | 6.13 | 6.12 | 8.65 |
| device timing(s) | 5.39 | 5.39 | 5.39 | 5.12 |


| memcpy | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 131072 | 131072 | 131072 | 131072 |
| host timing(s) | 4.3 | 13.2 | 4.7 | 1.94 |
| device timing(s) | 1.4 | 3.5 | 1.4 | 1.13 |


| miniFE | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2407 | NA | NA | 6645 |
| host timing(s) | 10.9 | NA | NA | NA | 15.5 |
| device timing(s) | 9.0 | NA | NA | 10.6 | 


| mixbench | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2049 |  2050 |  2050 | 6151 | 
| host timing(s) | 8.2 | 8.6 | 8.2 | 11.5 |
| device timing(s) | 7.52 | 7.5 | 7.47 | 7.56 |


| mkl-sgemm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 20004 | 60007 | 20007 | 20007 |
| host timing(s) | 1.63 | 14.6 | 19.1 | 6.7 |
| device timing(s) | 0.45 | 4.0 | 0.45 | 0.64 |


| murmurhash3 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 104 | 104 | 409 |
| host timing(s) | 10.3 | 10.9 | 11 | 13.5 |
| device timing(s) | 9.8 | 10.2 | 10.2 | 10.1 |


| nbody | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 402 | 402 | 402 | 1308 |
| host timing(s) | 4.1 | 4.3 | 3.1 | 7.3 |
| device timing(s) | 3.6 | 3.6 | 3.6 | 3.9 |


| nn | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 13 |
| host timing(s) | 0.2 | 0.55 | 0.59 | 3.5 |
| device timing(us) | 38 | 49 | 43 | 103 |


| nw | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2048 | 2050 | 2050 | 13314 |
| host timing(s) | 1.7 | 2.6 | 2.1 | 5.7 |
| device timing(s) | 0.57 | 0.88 | 0.76 | 1.47 |


| page-rank | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 6 | 11 | 11 | 30 |
| host timing(s) | 1.22 | 1.67 | 1.60 | 4.3 |
| device timing(s) | 0.71 | 0.77 | 0.74 | 0.82 |


| particle-diffusion | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 7 | 7 | 18 | 
| host timing(s) | 1.14 | 1.49 | 1.52 | 4.83 |
| device timing(s) | 0.24 | 0.48 | 0.41 | 1.32 |


| particlefilter | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 39 | 45 | 45 | 179 | 
| host timing(s) | 4.92 | 4.55 | 4.95 | 5.37 |
| device timing(s) | 4.86 | 4.48 | 4.87 | 5.02 |


| pathfinder | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 104 | 104 | 709 | 
| host timing(s) | 3.1 | 4.3 | 4.3 | 9.3 |
| device timing(s) | 2.8 | 3.6 | 3.6 | 5.8 |


| projectile | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 102 | 102 | 307 | 
| host timing(s) | 2.12 | 2.5 | 2.5 | 5.4 |
| device timing(s) | 1.75 | 1.75 | 1.74 | 1.82 |


| randomAccess | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 4 | 4 | 4 | 21 |
| host timing(s) | 1.94 | 2.2 | 2.3 | 6.4 | 
| device timing(s) | 1.45 | 1.45 | 1.45 | 2.8 |


| reduction | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 325 | 326 | 326 | 975 |
| host timing(s) | 1.61 | 1.93 | 1.94 | 4.8 |
| device timing(s) | 1.2 | 1.17 | 1.18 | 1.3 |


| reverse | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1048578 | 1048579 | 1048579 | 1048584 |
| host timing(s) | 31.3 | 28.9 | 57 | 173 |
| device timing(s) | 3.1 | 3.3 | 1.94 | 3.5 |


| rng-wallace | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 200 | 202 | 202 |  NA |
| host timing(s) | 2.6 | 3.2 | 3.2 | NA | 
| device timing(s) | 2.1 | 2.3 | 2.3 | NA |


| rsbench | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 10 | 10 | 34 |
| host timing(s) | 16.3 | 17.2 | 16.9 | 24.7 |
| device timing(s) | 14.3 | 14.7 | 14.3 | 19.1 |


| rtm8 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 32 | 38 | 38 | 220 |
| host timing(s) | 3.7 | 4.0 | 4.1 | 7.3 |
| device timing(s) | 3.1 | 3.1 | 3.2 | 3.6 |


| scan | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100001 | 10002 | 10002 | 200007 |
| host timing(s) | 34 | 33 | 31 | 50.3 |
| device timing(s) | 0.79 | 1.28 | 0.93 | 2.6 |


| softmax | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 |  507 |
| host timing(s) | 5.3 | 4.6 | 5.7 | 7.4 |
| device timing(s) | 4.9 | 3.9 | 5.1 | 3.9 |


| sph | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2002 | 2004 | 2004 | 13512 |
| host timing(s) | 21.8 | 22.3 | 22.2 | 24.8 |
| device timing(s) | 21.1 | 21 | 21.1 | 20.4 |


| srad | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 8003 | 8008 | 8008 | 36026 |
| host timing(s) | 2.2 | 2.75 | 2.2 | 6.4 |
| device timing(s) | 0.74 | 0.94 | 0.78 | 0.84 |


| sssp | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 88355 | 88641 | 88640 | NA |
| host timing(s) | 18.0 | 24.5 | 24.7 | NA |
| device timing(s) | 2.6 | 2.3 | 2.3 | NA |


| stencil | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 3 | 3 | 10 |
| host timing(s) | 0.68 | 0.89 | 0.98 | 4 |
| device timing(s) | 0.09 | 0.14 | 0.13 | 0.48 |


| su3 | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 104 | 104 | 715 |
| host timing(s) | 6.8 | 6.7 | 6.7 | 10.1 |
| device timing(s) | 6.3 | 5.8 | 5.8 | 6.5 |


| xsbench | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 9 | 9 | 26 |
| host timing(s) | 2.49 | 2.7 | 2.8 | 5.8 |
| device timing(s) | 2.1 | 2.0 | 2.0 | 2.3 |

# Reference
### all-pairs-distance
  All-pairs distance calculation (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2910913/)

### amgmk 
  The relax kernel in the AMGmk benchmark (https://asc.llnl.gov/CORAL-benchmarks/Micro/amgmk-v1.0.tar.gz)
  
### aobench
  A lightweight ambient occlusion renderer (https://code.google.com/archive/p/aobench)

### atomicIntrinsics
  Atomic add, subtract, min, max, AND, OR, XOR (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### backprop
  Backpropagation in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bezier-surface
  The Bezier surface (https://github.com/chai-benchmarks/chai)

### bfs
  The breadth-first search in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bitonic-sort
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### black-scholes
  The Black Scholes simulation (https://github.com/cavazos-lab/FinanceBench)

### bspline-vgh
  Bspline value gradient hessian (https://github.com/QMCPACK/miniqmc/blob/OMP_offload/src/OpenMP/main.cpp)

### b+tree
  B+Tree in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### cfd
  The CFD solver in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### chi2
  The Chi-square 2-df test. The input data must be downloaded before running the test. Please see README for the link.

### clenergy
  Direct Coulomb summation kernel (http://www.ks.uiuc.edu/Training/Workshop/GPU_Aug2010/resources/clenergy.tar.gz)

### clink
  Compact LSTM inference kernel (http://github.com/UCLA-VAST/CLINK)

### cobahh
  Simulation of Random Network of Hodgkin and Huxley Neurons with Exponential Synaptic Conductances (https://dl.acm.org/doi/10.1145/3307339.3343460)

### compute-score
  Document filtering (https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/compute-score.html)

### filter
  Filtering by a predicate (https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

### fft
  FFT in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### gamma-correction
  Gamma correction (https://github.com/intel/BaseKit-code-samples)

### gaussian
  Gaussian elimination in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### geodesic
  Geodesic distance (https://www.osti.gov/servlets/purl/1576565)

### haccmk
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

### heartwall
  Heart Wall in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### heat
  The heat equation solver (https://github.com/UoB-HPC/heat_sycl)

### heat2d
  Discreet 2D laplacian operation a number of times on a given vector (https://github.com/gpucw/cuda-lapl)

### histogram
  Histogram (http://github.com/NVlabs/cub/tree/master/experimental)

### hmm
  Hidden markov model (http://developer.download.nvidia.com/compute/DevZone/OpenCL/Projects/oclHiddenMarkovModel.tar.gz)

### hotspot3D
  hotspot3D in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### hybridsort
  hybridsort in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### inversek2j
  The inverse kinematics for 2-joint arm (http://axbench.org/)

### ising
  Monte Carlo simulations of 2D Ising Model (https://github.com/NVIDIA/ising-gpu/)

### iso2dfd, mandelbrot, particle-diffusion
  The HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### keccaktreehash 
  A Keccak tree hash function (http://sites.google.com/site/keccaktreegpu/)

### kmeans 
  K-means in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### knn
  K-nearest neighbor (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks/knn/)

### laplace
  A Laplace solver using red-black Gaussian Seidel with SOR solver (https://github.com/kyleniemeyer/laplace_gpu)

### lavaMD
  LavaMD in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lud
  LU decomposition in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### matrix-mul
  Single-precision floating-point matrix multiply

### matrix-rotate
  In-place matrix rotation

### maxpool3d
  3D Maxpooling (https://github.com/nachiket/papaa-opencl)

### md
  Molecular dynamics function in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### md5hash
  The MD5 Hash function (https://github.com/vetter/shoc/)

### memcpy
  A benchmark for memory copy from a host to a device

### miniFE
  MiniFE Mantevo mini-application (https://github.com/Mantevo/miniFE)

### mixbench
  A read-only version of mixbench (https://github.com/ekondis/mixbench)

### mkl-sgemm
  Single-precision floating-point matrix multiply using Intel<sup>®</sup> Math Kernel Library 

### murmurhash3
  MurmurHash3 yields a 128-bit hash value (https://github.com/aappleby/smhasher/wiki/MurmurHash3)

### nbody
  Nbody simulation (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/N-BodyMethods/Nbody)

### nn
  Needleman-Wunsch in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nw
  Needleman-Wunsch in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### page-rank
  PageRank (https://github.com/Sable/Ostrich/tree/master/map-reduce/page-rank)

### particlefilter
  Particle Filter in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### pathfinder
  PathFinder in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### projectile
  Projectile motion is a program that implements a ballistic equation (https://github.com/intel/BaseKit-code-samples)

### randomAccess
  Random memory access (https://icl.cs.utk.edu/projectsfiles/hpcc/RandomAccess/)

### reduction
  Integer sum reduction (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/reduction)

### reverse
   Reverse an input array of size 256 using shared memory

### rng-wallace
   Random number generation using the Wallace algorithm (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)

### rsbench
  A proxy application for full neutron transport application like OpenMC that support multipole cross section representations
  (https://github.com/ANL-CESAR/RSBench/)

### rtm8
  A structured-grid applications in the oil and gas industry (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/rtm8)

### scan
  A block-level scan using shared memory (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

### softmax
  The softmax function (https://github.com/pytorch/glow/tree/master/lib/Backends/OpenCL)

### sph
  The simple n^2 SPH simulation (https://github.com/olcf/SPH_Simple)

### srad
  SRAD (version 1) in the Rodinia Benchmark (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### sssp
  The single-source shortest path (https://github.com/chai-benchmarks/chai)

### stencil
  1D stencil using shared memory

### su3
  Lattice QCD SU(3) matrix-matrix multiply microbenchmark (https://gitlab.com/NERSC/nersc-proxies/su3_bench)

### xsbench
  A proxy application for full neutron transport application like OpenMC
  (https://github.com/ANL-CESAR/XSBench/)


## Development Team
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
Results presented were obtained using the Chameleon testbed supported by the National Science Foundation and the Intel<sup>®</sup> DevCloud.
