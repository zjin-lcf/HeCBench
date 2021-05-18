# oneAPI Direct Programming
This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA, HIP, SYCL, and OpenMP-4.5 target offloading. Intel<sup>®</sup> DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program in which memory management migration is implemented using the explicit and restricted Unified Shared Memory extension (DPCT usm) or the DPCT header files (DPCT header).


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

### boxfilter
  Box filtering (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

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

### convolutionSeperable 
  Convolution filter of a 2D image with separable kernels (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### d2q9_bgk
  A lattice boltzmann scheme with a 2D grid, 9 velocities, and Bhatnagar-Gross-Krook collision step (https://github.com/WSJHawkins/ExploringSycl)

### dct8x8
  Discrete Cosine Transform (DCT) and inverse DCT for 8x8 blocks (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### diamond
  Mask sequences kernel in Diamond (https://github.com/bbuchfink/diamond)

### divergence
  CPU and GPU divergence test (https://github.com/E3SM-Project/divergence_cmdvse)

### dp
  Dot product (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### dslash
  A Lattice QCD Dslash operator proxy application derived from MILC (https://gitlab.com/NERSC/nersc-proxies/milc-dslash)

### dxtc1
  DXT1 compression (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### easyWave
  Simulation of tsunami generation and propagation in the context of early warning (https://gitext.gfz-potsdam.de/geoperil/easyWave)

### extend2
  Smith-Waterman (SW) extension in Burrow-wheeler aligner for short-read alignment (https://github.com/lh3/bwa)

### extrema
  Find local maxima (https://github.com/rapidsai/cusignal/)

### fdtd3d
  FDTD-3D (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

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

### libor
  a LIBOR market model Monte Carlo application (https://people.maths.ox.ac.uk/~gilesm/cuda_old.html)

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

### medianfilter
  2-dimensional 3x3 Median Filter of RGBA image (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
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

### nms
  Work-efficient Parallel Non-Maximum Suppression Kernels (https://github.com/hertasecurity/gpu-nms)

### nn
  Nearest neighbor in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nw
  Needleman-Wunsch in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### page-rank
  PageRank (https://github.com/Sable/Ostrich/tree/master/map-reduce/page-rank)

### particlefilter
  Particle Filter in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### particles
  Particles collision simulation (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

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

### qrg
  Quasirandom generator implements Niederreiter quasirandom number generator and Moro's Inverse Cumulative Normal Distribution generator (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### radixsort
  A parallel radix sort (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
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

### secp256k1
  Part of BIP39 solver (https://github.com/johncantrell97/bip39-solver-gpu)

### shuffle
  shuffle instructions with subgroup sizes of 8, 16, and 32 (https://github.com/cpc/hipcl/tree/master/samples/4_shfl)

### simplemoc
  The attentuation of neutron fluxes across an individual geometrical segment (https://github.com/ANL-CESAR/SimpleMOC-kernel)

### snake
  Genome pre-alignment filtering (https://github.com/CMU-SAFARI/SneakySnake)

### sobol
  Sobol quasi-random generator (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### softmax
  The softmax function (https://github.com/pytorch/glow/tree/master/lib/Backends/OpenCL)

### sort
  Radix sort in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### sosfil
  Second-order IIR digital filtering (https://github.com/rapidsai/cusignal/)

### sph
  The simple n^2 SPH simulation (https://github.com/olcf/SPH_Simple)

### sptrsv
  A Thread-Level Synchronization-Free Sparse Triangular Solve (https://github.com/JiyaSu/CapelliniSpTRSV)

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

### testSNAP
  A proxy for the SNAP force calculation in the LAMMPS molecular dynamics package (https://github.com/FitSNAP/TestSNAP)

### thomas
  solve tridiagonal systems of equations using the Thomas algorithm (https://pm.bsc.es/gitlab/run-math/cuThomasBatch/tree/master)

### transpose
  Tensor transposition (https://github.com/Jokeren/GPA-Benchmark/tree/master/ExaTENSOR)

### triad
  Triad in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### tridiagonal
  Matrix solvers for large number of small independent tridiagonal linear systems(http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
### vmc
  Computes expectation values (6D integrals) associated with the helium atom (https://github.com/wadejong/Summer-School-Materials/tree/master/Examples/vmc)

### xsbench
  A proxy application for full neutron transport application like OpenMC
  (https://github.com/ANL-CESAR/XSBench/)


## Development Team
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
I would like to thank people who answered my questions about their codes: 
Bert de Jong, David Oro, Ian Karlin, Istvan Reguly, Jason Lau, Jeff Hammond, Jiya su, John Tramm, Mike Giles, Mohammed Alser, Nevin Liber, Pedro Valero Lara, Piotr Różański, Robert Harrison, Thomas Applencourt, Usman Roshan, Ye Luo, Yongbin Gu, Zhe Chen 


Results presented were obtained using the Chameleon testbed supported by the National Science Foundation and the Intel<sup>®</sup> DevCloud. The project used resources at [JLSE](https://www.jlse.anl.gov/) and [ExCL](https://excl.ornl.gov/). 	
