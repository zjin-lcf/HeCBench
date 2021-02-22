# oneAPI Direct Programming
This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA, SYCL, and OpenMP-4.5 target offloading. Intel<sup>®</sup> DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program in which memory management migration is implemented using the explicit and restricted Unified Shared Memory extension (DPCT usm) or the DPCT header files (DPCT header).


# Experiments
We compare the performance of the SYCL, DPCT-generated, and OpenMP implementations of each program on Intel integrated GPUs. The performance results below were obtained with the [Intel OpenCL intercept layer](https://github.com/intel/opencl-intercept-layer). "total enqueue" indicates the total number of low-level OpenCL enqueue commands called by a parallel program. These enqueue commands include "clEnqueueNDRangeKernel", "clEnqueueReadBuffer", and "clEnqueueWriteBuffer". The host timing is the total elapsed time of executing OpenCL API functions on a CPU host while the device timing is the total elapsed time of executing OpenCL API functions on a GPU device. The Plugin Interface is OpenCL.  
 
## Setup
Software: Intel<sup>®</sup> oneAPI Beta08* Toolkit, Ubuntu 18.04  
Platform 1: Intel<sup>®</sup> Xeon E3-1284L with a Gen8 P6300 integrated GPU  
Platform 2: Intel<sup>®</sup> Xeon E-2176G with a Gen9.5 UHD630 integrated GPU

*newer versions are only used when they can produce results correctly

## Run
A script "run.sh" attempts to run all tests with the OpenCL plugin interface. To run a single test, go to a test directory and type the command "make run".  

## Note
We may build the [software from source with support for Nvidia CUDA](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md). To run a single test on an Nvidia GPU, go to a test directory (e.g. affine-sycl) and type the command " make -f Makefile.cuda run". "Makefile.cuda" may be modified for different versions of Nvidia GPUs and toolchains.

We may build the [software from source with support for AMD HIP](https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md). To run a single test on an AMD GPU, go to a test directory (e.g. affine-sycl) and type the command " make -f Makefile.hip run". "Makefile.hip" may be modified for different versions of AMD GPUs and toolchains.

## Results on Platform 1
| affine | SYCL | DPCT usm | DPCT header | OpenMP |     
| --- | --- | --- | --- | --- |            
| total enqueue | 101 | 102 | 102 | 507 | 
| host timing(s) | 0.25 | 0.55 | 0.58 | 3.46 | 
| device timing(ms) | 6.7 | 6.8 | 6.6 | 8.1 |  


| all-pairs-distance | SYCL | DPCT usm | DPCT header | OpenMP |     
| --- | --- | --- | --- | --- |            
| total enqueue | 60 | 61 | 61 | 67 | 
| host timing(s) | 0.38 | 0.76 | 0.76 | 25.6 | 
| device timing(s) | 0.11 | 0.16 | 0.16 | 22 |  


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


| asta | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 550 | 550 | 550 | 1105 |
| host timing(s) | 5.0 | 5.7 | 5.7 | 16.2 | 
| device timing(s) | 4.7 | 5.2 | 5.1 | 12.6 |  


| atomicIntrinsics | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 3 | 3 | 3 | NA |
| host timing(s) | 9.3 | 9.7 | 9.7 | NA | 
| device timing(s) | 9.1 | 9.1 | 9.1 | NA |  


| axhelm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 105 | 105 | NA | 
| host timing(s) | 8.15/14.6 | 9.1/15.9 | 9.2/16.0 | NA | 
| device timing(s) | 4.4/10.8 | 5.0/11.8 | 5.0/11.7 | NA |  


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


| bsearch | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 80 |  160 |  160 | 308 | 
| host timing(s) | 2.28 | 2.43 | 2.43 | 2.73 |
| device timing(s) | 2.11 | 2.19 | 2.17 | 2.25 |


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


| ccsd-trpdrv | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 500 | 2400 | 2400 | 3405 |
| host timing(s) | 7.9 | 8.3 | 8.2 | 12.7 |
| device timing(s) | 7.9 | 8.2 | 8.1 | 12.3 |


| ced | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 660 | 663 | 663 | NA |
| host timing(s) | NA | NA | NA | NA |
| device timing(ms) | NA | NA | NA | NA |


| cfd | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 16005 | NA | NA | 132035 |
| host timing(s) | 4.4 | 4.2 | NA | 9.95 |
| device timing(s) | 3.5 | 3.4 | NA | 3.76 |


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


| d2q9_bgk | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 80011 | 80021  | 80021 | 640027 |
| host timing(s) | 19.3 | 14.7 | 21.4 | 39.2 |
| device timing(s) | 14.4 | 15.4 | 16.0 | 20.8  |


| diamond | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 5  | 5 | 14 |
| host timing(s) | 41.6 | 41.9 | 42.3 | 43.8 |
| device timing(s) | 40.8 | 40.9 | 41.2 | 40.2  |


| divergence | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 40000 | 100000  | 100000 | 280006 |
| host timing(s) | 1.6 | 6.2 | 7.4 | 8.6 |
| device timing(s) | 0.33 | 0.42 | 0.42 | 0.34  |


| easyWave | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 17293 | 17299  | 17299 | 69183 |
| host timing(s) | 29.8 | 31.0 | 32.0 | 41.8 |
| device timing(s) | 28.5 | 29.3 | 29.9 | 36.6  |


| extend2 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 14000 | 24000  | 24000 | 46005 |
| host timing(s) | 10.9 | 11.5 | 11.8 | 20.2 |
| device timing(s) | 9.6 | 9.6 | 9.6 | 16.1 |


| extrema | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 9696 | 9792  | 9792 | 99404 |
| host timing(s) | 34.2 | 37.9 | 38.6 | 44.8 |
| device timing(s) | 33.5 | 36.8 | 37.2 | 39.1 |


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


| floydwarshall | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 51251 | 51251 | 51251 | 512056 |
| host timing(s) | 10.3 | 7.2 | 10.7 | 21.8 |
| device timing(ms) | 8.9 | 6.85 | 8.9 | 10.4 | 


| fpc | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 400 | 800 | 800 | NA |
| host timing(s) | 5.1 | 4.1 | 4.2 | NA |
| device timing(ms) | 0.64 | 1.2 | 1.2 | NA | 


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


| gmm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 20287 | 20291 | NA | 44753 |
| host timing(s) | 2.7 | 3.6 | NA | 7.0 |
| device timing(s) | 1.13 | 1.92 | NA | 2.35 |


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


| interleave | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 202 | 206 | 206 | 1012 |
| host timing(s) | 13.4 | 12.0 | 13.9 | 14.9 | 
| device timing(s) | 13.1 | 11.4 | 13.3 | 3.1 |


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


| jaccard | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 108 | 108 | NA |
| host timing(s) | 28.6  | 28.8 | 28.9 | NA |
| device timing(s) | 28.2 | 28.1 | 28.1 | NA |


| jenkins-hash | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 104 | 104 | 509 |
| host timing(s) | 6.6  | 7.2 | 7.2 | 9.9 |
| device timing(s) | 6.3 | 6.4 | 6.4 | 6.3 |


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


| lanczos | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 9108 | 9156 | 9156 | 37620 |
| host timing(s) | 16.0 | 17.9 | 18.3 |  2642 |
| device timing(s) | 14.9 | 16.7 | 16.8 | 2637  |


| langford | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 15 | 20 | 20 | 33 |
| host timing(s) | 7.68 | 8.1 | 8.1 | 10.2 |
| device timing(s) | 5.8 | 5.8 | 5.8 | 5.3 |


| laplace | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 12742 | 12751 | 12751 | 38237 |
| host timing(s) | 1.18 | 1.27 | 1.66 | 4.95 |
| device timing(s) | 0.28 | 0.28 | 0.27 | 0.67 |


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


| lid-driven-cavity | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1605667 | 1605673 | 1605673 | 5619820 | 
| host timing(s) | 216 | 276| 263 | 375 |
| device timing(s) | 154 | 212 | 228 | 223 |


| lombscargle | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 104 | 104 | 309 |
| host timing(s) | 10.4 | 10.7 | 10.7 | 13.6 |
| device timing(s) | 10.0 | 10.0 | 10.0 | 10.0 |


| lud | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1535 | 1536 | 1536 | 6145 |
| host timing(s) | 8.9 | 11.0 | 11.2 | 14.1 |
| device timing(s) | 7.8 | 9.7 | 9.8 | 9.7 |


| lulesh | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2970 | 2986 | 2986 | 6635 |
| host timing(s) | 35.8 | 30.1 | 35.9 | 33.5 |
| device timing(s) | 32.0 | 27.4 | 31.5 | 28.1 |


| memcpy | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 131072 | 131072 | 131072 | 131072 |
| host timing(s) | 4.3 | 4.9 | 4.6 | 2.4 |
| device timing(s) | 1.2 | 1.5 | 1.2 | 1.7 |


| miniFE | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2404 | 2412 | 2412 | 6645 |
| host timing(s) | 9.5 | 9.8 | 10.0 | 16.7 |
| device timing(s) | 8.7 | 8.8 | 8.8 | 12.8 | 


| minimap2 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | NA | 20 | NA | 83 |
| host timing(s) | NA | 1.95 | NA | 4.86|
| device timing(s) | NA | 1.14 | NA | 1.29 |


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


| mt | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 203 | 204 | 204 | 1018 | 
| host timing(s) | 1.47 | 1.75 | 1.80 | 5.4 |
| device timing(s) | 1.06 | 1.06 | 1.06 | 1.72 |


| multimaterial | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100 | 429 | 429 | 689 |
| host timing(s) | 3.2 | 3.5 | 3.5 | 6.8 |
| device timing(s) | 0.88 | 1.6 | 1.6 | 2.95 |


| murmurhash3 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 104 | 104 | 409 |
| host timing(s) | 6.3 | 7.4 | 7.2 | 20.7 |
| device timing(s) | 5.9 | 6.7 | 6.4 | 17.0 |


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


| popcount | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 505 | 506 | 506 | 2015 |
| host timing(s) | 6.03 | 6.37 | 6.43 | 9.5 |
| device timing(s) | 5.7 | 5.7 | 5.7 | 5.9 |


| present | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 200 | NA | NA | 512 | 
| host timing(s) | 1.46 | NA | NA | 4.36 |
| device timing(s) | 0.94 | NA | NA | 0.67 |


| projectile | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 102 | 102 | 307 | 
| host timing(s) | 3.0 | 3.4 | 3.5 | 6.5 |
| device timing(s) | 2.7 | 2.7 | 2.7 | 2.85 |


| quicksort | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2250 | 3670 | NA | NA | 
| host timing(s) | 9.3 | 16.9 | NA | NA | 
| device timing(s) | 2.1 | 3.4 | NA | NA |


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
| total enqueue | 200 | 202 | 202 |  207 |
| host timing(s) | 4.3 | 3.9 | 4.6 | 6.6 | 
| device timing(s) | 3.6 | 3.2 | 3.6 | 3.1 |


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


| s3d | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2701 |  2705 | 5410 | 34441 |
| host timing(s) | 21 | 161 | 323 | 13 |
| device timing(s) | 0.24 | 0.29 | 1.45 | 0.42 |


| scan | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100001 | 10002 | 10002 | 200007 |
| host timing(s) | 3.4 | 2.9 | 3.8 | 10.3 |
| device timing(s) | 0.69 | 1.22 | 0.9 | 1.39 |


| shuffle | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 909 | 910 | 910 | NA |
| host timing(s) | 28.7 | 26.2 | 29.1 | NA |
| device timing(s) | 28.2 | 25.4 | 28.1 | NA |


| simplemoc | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 12 | 18 | 18 | 73 |
| host timing(s) | 14.3 | 40.2 | 16.3 | 36.0 |
| device timing(s) | 14.1 | 39.6 | 15.6 | 32.4 |


| softmax | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 |  507 |
| host timing(s) | 1.6 | 4.5 | 1.9 | 5.1 |
| device timing(s) | 1.3 | 3.8 | 1.3 | 1.5 |


| sort | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 241 | 242 | 242 | NA |
| host timing(s) | 4.94 | 22.6 | 22.6 | NA |
| device timing(s) | 4.49 | 21.8 | 21.8 | NA |


| sosfil | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 202 | 208 | 208 | 1214 |
| host timing(s) | 1.82 | 2.01 | 2.03 | 2.59 |
| device timing(s) | 1.79 | 1.95 | 1.96 | 2.24 |


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


| streamcluster | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 11278 | 11278 | 11278 | 30617 |
| host timing(s) | 4.5 | 4.7 | 4.9 | 8.5 |
| device timing(s) | 3.6 | 3.7 | 3.6 | 4.4 |


| su3 | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 104 | 104 | 715 |
| host timing(s) | 7.9 | 8.2 | 8.3 | 11 |
| device timing(s) | 7.4 | 7.4 | 7.4 | 7.4 |


| thomas | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 105 | 105  | 710 |
| host timing(s) | 5.8 | 2.8 | 6.2 | 9.1 |
| device timing(s) | 5.4 | 2.2 | 5.5 | 5.6 |


| transpose | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 51 | NA | 59  | 64 |
| host timing(s) | 12.9 | NA | 13.6 | 25.8 |
| device timing(s) | 11.9 | NA | 12.4 | 22 |


| triad | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 204400 | 204400 | 204400  | 407907 |
| host timing(s) | 7.2 | 7.3 | 7.4 | 98 |
| device timing(s) | 3.4 | 2.8 | 3.4 | 86 |


| xsbench | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 9 | 9 | 26 |
| host timing(s) | 3.0 | 3.4 | 3.0 | 6.8 |
| device timing(s) | 2.6 | 2.7 | 2.1 | 3.1 |

## Results on Platform 2
| affine | SYCL | DPCT usm | DPCT header | OpenMP |     
| --- | --- | --- | --- | --- |            
| total enqueue | 101 | 102 | 102 | 507 | 
| host timing(s) | 0.33 | 0.62 | 0.62 | 3.1 | 
| device timing(ms) | 11.5 | 12.7 | 11.0 | 12.4 |  


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


| asta | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 550 | 550 | 550 | 1105 |
| host timing(s) | 5.6 | 6.1 | 6.1 | 15.8 | 
| device timing(s) | 5.4 | 5.8 | 5.7 | 12.9 |  


| atomicIntrinsics | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 3 | 3 | 3 | NA |
| host timing(s) | 1.0 | 1.37 | 1.35 | NA | 
| device timing(s) | 0.73 | 0.73 | 0.73 | NA |  


| axhelm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 105 | 105 | NA | 
| host timing(s) | 6.3/9.6 | 7.1/10.7 | 7.1/10.2 | NA | 
| device timing(s) | 3.2/6.6 | 3.5/6.8 | 3.5/6.8 | NA |  


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


| bsearch | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 80 |  160 |  160 | 308 | 
| host timing(s) | 2.3 | 2.5 | 2.5 | 2.5 |
| device timing(s) | 2.2 | 2.3 | 2.3 | 2.1 |


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


| ccsd-trpdrv | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 500 | 2400 | 2400 | 3405 |
| host timing(s) | 14.9 | 18.5 | 19.3 | 15.2 |
| device timing(s) | 13.7 | 14.2 | 15.9 | 11.0 |


| ced | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 660 | 663 | 663 | 1548 |
| host timing(s) | 0.43 | 0.75 | 0.84 | 3.75 |
| device timing(ms) | 44 | 49 | 49  | 60 |



| cfd | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 16005 | 16013 | 16013 | 132035 |
| host timing(s) | 3.75 | 8.2 | 4.9 | 26.5 |
| device timing(s) | 3.04 | 3.0 | 3.02 | 18.1 |


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


| d2q9_bgk | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 80011 | 80021  | 80021 | 640027 |
| host timing(s) | 16.4 | 12.9 | 16.7 | 100.1 |
| device timing(s) | 12.7 | 13.0 | 12.9 | 51  |


| diamond | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 5  | 5 | 14 |
| host timing(s) | 26.1 | 26.7 | 26.1 | 29.4 |
| device timing(s) | 25.4 | 25.7 | 25.1 | 25.9  |


| divergence | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 40000 | 100000  | 100000 | 280006 |
| host timing(s) | 13.2 | 48.6 | 69.8 | 63.3 |
| device timing(s) | 0.71 | 0.41 | 0.44 | 0.33  |


| easyWave | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 17293 | 17299  | 17299 | 69183 |
| host timing(s) | 29.6 | 32.9 | 32.3 | 38.3 |
| device timing(s) | 28.1 | 27.8 | 27.8 | 32.8  |


| extend2 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 14000 | 24000  | 24000 | 46005 |
| host timing(s) | 14.9 | 16.1 | 15.5 | 19.8 |
| device timing(s) | 7.7 | 7.8 | 7.8 | 13.7 |


| extrema | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 9696 | 9792  | 9792 | 99404 |
| host timing(s) | 59.4 | 49.4 | 58.5 | 95.1 |
| device timing(s) | 55.5 | 50.1 | 55.0 | 91.8 |


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


| floydwarshall | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 51251 | 51251 | 51251 | 512056 |
| host timing(s) | 22.4 | 24.4 | 20.4 | 138 |
| device timing(ms) | 7.2 | 6.7 | 7.2 | 23.3 | 


| fpc | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 400 | 800 | 800 | NA |
| host timing(s) | 3.9 | 4.4 | 4.3 | NA |
| device timing(ms) | 0.74 | 1.6 | 1.6 | NA | 


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


| gmm | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 20287 | 20291 | NA | 44753  |
| host timing(s) | 130 | 450 | NA | 222 |
| device timing(s) | 1.1 | 2.8 | NA | 15.3 |


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


| histogram | SYCL | DPCT usm | DPCT header | OpenMP* (to be optimized) | 
| --- | --- | --- | --- | --- |
| total enqueue | 1218 | 1221 | 1221 | 3666 |
| host timing(s) | 2.75 | 2.5 | 2.6 | 15.5 |
| device timing(s) | 0.83 | 0.86 | 0.85 | 58 |


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
| host timing(s) | 1.68 | 1.86 | 1.91 | 4.66 |
| device timing(s) | 1.21 | 1.0 | 1.03 | 1.44 |


| interleave | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 202 | 206 | 206 | 1012 |
| host timing(s) | 20.8 | 20.6 | 20.0 | 23.9 | 
| device timing(s) | 20.5 | 19.8 | 19.3 | 20.2 |


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


| jaccard | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 108 | 108 | NA |
| host timing(s) | 41.7  | 42.1 | 41.2 | NA |
| device timing(s) | 41.3 | 41.7 | 40.6 | NA |


| jenkins-hash | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 104 | 104 | 509 |
| host timing(s) | 4.9  | 5.4 | 5.6 | 8.1 |
| device timing(s) | 4.6 | 4.6 | 4.8 | 4.9 |


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


| lanczos | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 9108 | 9156 | 9156 | 37620 |
| host timing(s) | 37.3 | 40.4 | 26.7 | 82.8
| device timing(s) | 32.2 | 37.9 | 21.3 | 80.9 |


| langford | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 15 | 20 | 20 | 33 |
| host timing(s) | 12.9 | 13.0 | 12.8 | 37.4 |
| device timing(s) | 11.3 | 11.3 | 11.1 | 33 |


| laplace | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 12742 | 12751 | 12751 | 38237 |
| host timing(s) | 9.98 | 7.7 | 10.4 | 16.4 |
| device timing(s) | 0.65 | 0.94 | 0.66 | 3.56 |


| lavaMD | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 6 | 6 | 15 |
| host timing(s) | 1.8 | 2.0 | 2.0 | 4.8 |
| device timing(s) | 1.42 | 1.31 | 1.27 | 1.32 |


| leukocyte | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 204 | 711 | 711 | 1334 |
| host timing(s) | 4.1 | 4.6| 4.6 | 6.8 |
| device timing(s) | 3.58 | 3.77 | 3.73 | 3.64 |


| lid-driven-cavity | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 1605667 | 1605673 | 1605673 | 5619820 | 
| host timing(s) | 264 | 573| 289 | 712 |
| device timing(s) | 201 | 222 | 220 | 231 |


| lombscargle | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 104 | 104 | 309 |
| host timing(s) | 2.4 | 2.96 | 2.91 | 5.6 |
| device timing(s) | 1.95 | 2.14 | 2.12 | 2.1 |


| lulesh | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2970 | 2986 | 2986 | 6635 |
| host timing(s) | 46.9 | 47.3 | 48.8 | 137.9 |
| device timing(s) | 41.6 | 43.6 | 43.4 | 132 |


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
| total enqueue | 2404 | 2412 | 2412 | 6638 |
| host timing(s) | 12.7 | 10.3 | 21.4 | 28.7 |
| device timing(s) | 11.1 | 9.2 | 19.2 | 23.4 | 


| minimap2 | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | NA | 20 | NA | 83 |
| host timing(s) | NA | 1.63 | NA | 9.26 |
| device timing(s) | NA | 0.93 | NA | 6.09 |


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


| mt | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 203 | 204 | 204 | 1018 | 
| host timing(s) | 1.48 | 1.42 | 1.57 | 10.8 |
| device timing(s) | 1.06 | 1.02 | 1.05 | 7.6 |


| multimaterial | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100 | 429 | 429 | 689 |
| host timing(s) | 3.3 | 3.4 | 4.5 | 7.8 |
| device timing(s) | 1.6 | 1.8 | 2.6 | 4.2 |


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


| popcount | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 505 | 506 | 506 | 2015 |
| host timing(s) | 8.8 | 9.4 | 8.9 | 12.5 |
| device timing(s) | 8.4 | 8.2 | 8.2 | 8.9 |


| present | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 200 | NA | NA | 512 | 
| host timing(s) | 1.37 | NA | NA | 6.1 |
| device timing(s) | 0.97 | NA | NA | 2.78 |


| projectile | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 102 | 102 | 307 | 
| host timing(s) | 2.12 | 2.5 | 2.5 | 5.4 |
| device timing(s) | 1.75 | 1.75 | 1.74 | 1.82 |


| quicksort | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2184 | 3685 | 3700 | 6696 | 
| host timing(s) | 15.4 | 21.7 | 22.4 | 23.7 | 
| device timing(s) | 8.4 | 9.4 | 9.45 | 151.3 |


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
| total enqueue | 200 | 202 | 202 |  207 |
| host timing(s) | 2.6 | 3.2 | 3.2 | 15.0 | 
| device timing(s) | 2.1 | 2.3 | 2.3 | 11.7 |


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


| s3d | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 2701 |  2705 | 5410 | 34441 |
| host timing(s) | 9.1 | 29 | 64 | 7.1 |
| device timing(s) | 0.21 | 0.26 | 0.22 | 0.2 |


| scan | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 100001 | 10002 | 10002 | 200007 |
| host timing(s) | 34 | 33 | 31 | 50.3 |
| device timing(s) | 0.79 | 1.28 | 0.93 | 2.6 |


| shuffle | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 909 | 910 | 910 | NA |
| host timing(s) | 25.3 | 25.3 | 25.4 | NA |
| device timing(s) | 24.4 | 24.5 | 24.4 | NA |


| simplemoc | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 12 | 18 | 18 | 73 |
| host timing(s) | 46.1 | 47.5 | 43.8 | 50.3 |
| device timing(s) | 45.6 | 46.7 | 43.1 | 47 |


| snake | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 5202 | 5202 | 5202 |  20807 |
| host timing(s) | 14.6 | 36.7 | 14.8 | 44.7 |
| device timing(s) | 11.2 | 34.3 | 11.2 | 36.9 |


| softmax | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 101 | 102 | 102 |  507 |
| host timing(s) | 5.3 | 4.6 | 5.7 | 7.4 |
| device timing(s) | 4.9 | 3.9 | 5.1 | 3.9 |


| sort | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 241 | 242 | 242 | 256 |
| host timing(s) | 10.2 | 17.2 | 17.9 | 26.6 |
| device timing(s) | 9.6 | 16.6 | 17.3 | 23.1 |


| sosfil | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 202 | 208 | 208 | NA |
| host timing(s) | 1.78 | 1.95 | 1.95 | NA |
| device timing(s) | 1.73 | 1.91 | 1.89 | NA |


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


| streamcluster | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 11278 | 11278 | 11278 | 30617 |
| host timing(s) | 7.8 | 9.0 | 7.7 | 11.9 |
| device timing(s) | 5.95 | 6.1 | 5.95 | 6.6 |


| su3 | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 104 | 104 | 715 |
| host timing(s) | 6.8 | 6.7 | 6.7 | 10.1 |
| device timing(s) | 6.3 | 5.8 | 5.8 | 6.5 |


| thomas | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 102 | 105 | 105  | 710 |
| host timing(s) | 4.6 | 2.6 | 4.7 | 10.4 |
| device timing(s) | 4.2 | 2.2 | 4.2 | 7.5 |


| transpose | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 51 | NA | 59  | 64 |
| host timing(s) | 12.3 | NA | 12.5 | 17.9 |
| device timing(s) | 11.1 | NA | 11.3 | 14.1 |


| triad | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 204400 | 204400 | 204400  | 407907 |
| host timing(s) | 7.9 | 8.8 | 8.4 | 192 |
| device timing(s) | 3.8 | 3.3 | 3.8 | 96 |


| xsbench | SYCL | DPCT usm | DPCT header | OpenMP |
| --- | --- | --- | --- | --- |
| total enqueue | 2 | 9 | 9 | 26 |
| host timing(s) | 2.49 | 2.7 | 2.8 | 5.8 |
| device timing(s) | 2.1 | 2.0 | 2.0 | 2.3 |


## Results on Platform 3
#### Intel<sup>®</sup> Core<sup>TM</sup> i9-10920X CPU with a Gen12LP discrete GPU (DG1)
| minimod | SYCL | DPCT usm | DPCT header | OpenMP | 
| --- | --- | --- | --- | --- |
| total enqueue | 8008 | 8011 | 8011 | NA |
| host timing(s) | 2.4 | 1.9 | 2.0 | NA | 
| device timing(s) | 0.69 | 0.68 | 0.68 | NA | 


# Reference
### affine
  Affine transformation (https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/affine)

### all-pairs-distance
  All-pairs distance calculation (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2910913/)

### amgmk 
  The relax kernel in the AMGmk benchmark (https://asc.llnl.gov/CORAL-benchmarks/Micro/amgmk-v1.0.tar.gz)
  
### aobench
  A lightweight ambient occlusion renderer (https://code.google.com/archive/p/aobench)

### asta
  Array of structure of tiled array for data layout transposition (https://github.com/chai-benchmarks/chai)

### atomicIntrinsics
  Atomic add, subtract, min, max, AND, OR, XOR (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### axhelm
  Helmholtz matrix-vector product (https://github.com/Nek5000/nekBench/tree/master/axhelm)

### backprop
  Backpropagation in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bezier-surface
  The Bezier surface (https://github.com/chai-benchmarks/chai)

### bfs
  The breadth-first search in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bitonic-sort
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### black-scholes
  The Black-Scholes simulation (https://github.com/cavazos-lab/FinanceBench)

### bsearch
  Classic and vectorizable binary search algorithms (https://www.sciencedirect.com/science/article/abs/pii/S0743731517302836)

### bspline-vgh
  Bspline value gradient hessian (https://github.com/QMCPACK/miniqmc/blob/OMP_offload/src/OpenMP/main.cpp)

### b+tree
  B+Tree in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### ccsd-trpdrv
  The CCSD tengy kernel, which was converted from Fortran to C by Jeff Hammond, in NWChem (https://github.com/jeffhammond/nwchem-ccsd-trpdrv)

### ced
  Canny edge detection (https://github.com/chai-benchmarks/chai)

### cfd
  The CFD solver in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### chi2
  The Chi-square 2-df test. The input data must be downloaded before running the test. Please see README for the link.

### clenergy
  Direct coulomb summation kernel (http://www.ks.uiuc.edu/Training/Workshop/GPU_Aug2010/resources/clenergy.tar.gz)

### clink
  Compact LSTM inference kernel (http://github.com/UCLA-VAST/CLINK)

### cobahh
  Simulation of Random Network of Hodgkin and Huxley Neurons with Exponential Synaptic Conductances (https://dl.acm.org/doi/10.1145/3307339.3343460)

### compute-score
  Document filtering (https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/compute-score.html)

### d2q9_bgk
  A lattice boltzmann scheme with a 2D grid, 9 velocities, and Bhatnagar-Gross-Krook collision step (https://github.com/WSJHawkins/ExploringSycl)

### diamond
  Mask sequences kernel in Diamond (https://github.com/bbuchfink/diamond)

### divergence
  CPU and GPU divergence test (https://github.com/E3SM-Project/divergence_cmdvse)

### easyWave
  Simulation of tsunami generation and propagation in the context of early warning (https://gitext.gfz-potsdam.de/geoperil/easyWave)

### extend2
  Smith-Waterman (SW) extension in Burrow-wheeler aligner for short-read alignment (https://github.com/lh3/bwa)

### extrema
  Find local maxima (https://github.com/rapidsai/cusignal/)

### filter
  Filtering by a predicate (https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

### fft
  FFT in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### floydwarshall
  Floyd-Warshall Pathfinding sample (https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/HIP-Examples-Applications/FloydWarshall/)

### fpc
  Frequent pattern compression ( Base-delta-immediate compression: practical data compression for on-chip caches. In Proceedings of the 21st international conference on Parallel architectures and compilation techniques (pp. 377-
388). ACM.)

### gamma-correction
  Gamma correction (https://github.com/intel/BaseKit-code-samples)

### gaussian
  Gaussian elimination in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### geodesic
  Geodesic distance (https://www.osti.gov/servlets/purl/1576565)

### gmm
  Expectation maximization with Gaussian mixture models (https://github.com/Corv/CUDA-GMM-MultiGPU)

### haccmk
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

### heartwall
  Heart Wall in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### heat
  The heat equation solver (https://github.com/UoB-HPC/heat_sycl)

### heat2d
  Discreet 2D laplacian operation a number of times on a given vector (https://github.com/gpucw/cuda-lapl)

### histogram
  Histogram (http://github.com/NVlabs/cub/tree/master/experimental)

### hmm
  Hidden markov model (http://developer.download.nvidia.com/compute/DevZone/OpenCL/Projects/oclHiddenMarkovModel.tar.gz)

### hotspot3D
  Hotspot3D in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### hybridsort
  Hybridsort in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### interleave
  Interleaved and non-interleaved global memory accesses (Shane Cook. 2012. CUDA Programming: A Developer's Guide to Parallel Computing with GPUs (1st. ed.). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.)

### inversek2j
  The inverse kinematics for 2-joint arm (http://axbench.org/)

### ising
  Monte-Carlo simulations of 2D Ising Model (https://github.com/NVIDIA/ising-gpu/)

### iso2dfd, mandelbrot, particle-diffusion
  The HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### jaccard
  Jaccard index for a sparse matrix (https://github.com/rapidsai/nvgraph/blob/main/cpp/src/jaccard_gpu.cu)

### jenkins-hash 
  Bob Jenkins lookup3 hash function (https://android.googlesource.com/platform/external/jenkins-hash/+/75dbeadebd95869dd623a29b720678c5c5c55630/lookup3.c)

### keccaktreehash 
  A Keccak tree hash function (http://sites.google.com/site/keccaktreegpu/)

### kmeans 
  K-means in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### knn
  K-nearest neighbor (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks/knn/)

### lanczos
  Lanczos tridiagonalization (https://github.com/linhr/15618)

### langford
  Count planar Langford sequences (https://github.com/boris-dimitrov/z4_planar_langford_multigpu)

### laplace
  A Laplace solver using red-black Gaussian Seidel with SOR solver (https://github.com/kyleniemeyer/laplace_gpu)

### lavaMD
  LavaMD in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### leukocyte 
  Leukocyte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lid-driven-cavity 
  GPU solver for a 2D lid-driven cavity problem (https://github.com/kyleniemeyer/lid-driven-cavity_gpu)

### lombscargle
   Lomb-Scargle periodogram (https://github.com/rapidsai/cusignal/)

### lud
  LU decomposition in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lulesh
  Livermore unstructured Lagrangian explicit shock hydrodynamics (https://github.com/LLNL/LULESH)

### matrix-mul
  Single-precision floating-point matrix multiply

### matrix-rotate
  In-place matrix rotation

### maxpool3d
  3D Maxpooling (https://github.com/nachiket/papaa-opencl)

### md
  Molecular dynamics function in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### md5hash
  MD5 Hash function in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### memcpy
  A benchmark for memory copy from a host to a device

### miniFE
  MiniFE Mantevo mini-application (https://github.com/Mantevo/miniFE)

### minimap2
  Hardware Acceleration of Long Read Pairwise Overlapping in Genome Sequencing (https://github.com/UCLA-VAST/minimap2-acceleration)

### minimod
  A finite difference solver for seismic modeling (https://github.com/rsrice/gpa-minimod-artifacts)

### mixbench
  A read-only version of mixbench (https://github.com/ekondis/mixbench)

### mkl-sgemm
  Single-precision floating-point matrix multiply using Intel<sup>®</sup> Math Kernel Library 

### mt
  Mersenne Twister (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### multimaterial
  Multi-material simulations (https://github.com/reguly/multimaterial)

### murmurhash3
  MurmurHash3 yields a 128-bit hash value (https://github.com/aappleby/smhasher/wiki/MurmurHash3)

### nbody
  Nbody simulation (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/N-BodyMethods/Nbody)

### nn
  Nearest neighbor in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nw
  Needleman-Wunsch in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### page-rank
  PageRank (https://github.com/Sable/Ostrich/tree/master/map-reduce/page-rank)

### particlefilter
  Particle Filter in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### pathfinder
  PathFinder in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### popcount
  Implementations of population count (Jin, Z. and Finkel, H., 2020, May. Population Count on Intel® CPU, GPU and FPGA. In 2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (pp. 432-439). IEEE.)

### present
  Lightweight cryptography (https://github.com/bozhu/PRESENT-C/blob/master/present.h)

### projectile
  Projectile motion is a program that implements a ballistic equation (https://github.com/intel/BaseKit-code-samples)

### quicksort
  Quicksort (https://software.intel.com/content/www/us/en/develop/download/code-for-the-parallel-universe-article-gpu-quicksort-from-opencl-to-data-parallel-c.html)

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

### s3d
  S3D in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### scan
  A block-level scan using shared memory (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

### shuffle
  shuffle instructions with subgroup sizes of 8, 16, and 32 (https://github.com/cpc/hipcl/tree/master/samples/4_shfl)

### simplemoc
  The attentuation of neutron fluxes across an individual geometrical segment (https://github.com/ANL-CESAR/SimpleMOC-kernel)

### snake
  Genome pre-alignment filtering (https://github.com/CMU-SAFARI/SneakySnake)

### softmax
  The softmax function (https://github.com/pytorch/glow/tree/master/lib/Backends/OpenCL)

### sort
  Radix sort in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### sosfil
  Second-order IIR digital filtering (https://github.com/rapidsai/cusignal/)

### sph
  The simple n^2 SPH simulation (https://github.com/olcf/SPH_Simple)

### srad
  SRAD (version 1) in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### sssp
  The single-source shortest path (https://github.com/chai-benchmarks/chai)

### stencil
  1D stencil using shared memory

### streamcluster
  Streamcluster in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### su3
  Lattice QCD SU(3) matrix-matrix multiply microbenchmark (https://gitlab.com/NERSC/nersc-proxies/su3_bench)

### thomas
  solve tridiagonal systems of equations using the Thomas algorithm (https://pm.bsc.es/gitlab/run-math/cuThomasBatch/tree/master)

### transpose
  Tensor transposition (https://github.com/Jokeren/GPA-Benchmark/tree/master/ExaTENSOR)

### triad
  Triad in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### xsbench
  A proxy application for full neutron transport application like OpenMC
  (https://github.com/ANL-CESAR/XSBench/)


## Development Team
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
Results presented were obtained using the Chameleon testbed supported by the National Science Foundation and the Intel<sup>®</sup> DevCloud. The project also uses resources at [JLSE](https://www.jlse.anl.gov/) and [ExCL](https://excl.ornl.gov/). 	
