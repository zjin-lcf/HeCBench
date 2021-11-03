# HeCBench
This repository contains a collection of Heterogeneous Computing benchmarks written with CUDA, HIP, SYCL (DPC++), and OpenMP-4.5 target offloading for studying performance, portability, and productivity. 

# Software installation
[AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)  
[Intel DPC++ compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) or [Intel oneAPI toolkit](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html)  
[Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)

# Dataset
For Rodinia benchmarks, please download the dataset at http://lava.cs.virginia.edu/Rodinia/download.htm 

# Known issues
The programs have not been evaluated on Windows or MacOS
The lastest Intel SYCL compiler (not the Intel oneAPI toolkit) may be needed for building some SYCL programs successfully  
Kernel results do not exactly match using these programming languages on a platform for certain programs  
Not all programs automate the verification of host and device results  
Not all CUDA programs have SYCL, HIP or OpenMP equivalents  
Not all programs have OpenMP target offloading implementations  
Raw performance of any program may be suboptimal  
Some programs may take longer to complete on an integrated GPU

# Experimental Results
Early results are shown [here](results/README.md)

# Reference
### ace
  Phase-field simulation of dendritic solidification (https://github.com/myousefi2016/Allen-Cahn-CUDA)

### adv
  Advection (https://github.com/Nek5000/nekBench/tree/master/adv)

### aes
  AES encrypt and decrypt (https://github.com/Multi2Sim/m2s-bench-amdsdk-2.5-src)

### affine
  Affine transformation (https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/affine)

### aidw
  Adaptive inverse distance weighting (Mei, G., Xu, N. & Xu, L. Improving GPU-accelerated adaptive IDW interpolation algorithm using fast kNN search. SpringerPlus 5, 1389 (2016))

### align-types
  Alignment specification for variables of structured types (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### all-pairs-distance
  All-pairs distance calculation (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2910913/)

### amgmk 
  The relax kernel in the AMGmk benchmark (https://asc.llnl.gov/CORAL-benchmarks/Micro/amgmk-v1.0.tar.gz)
  
### aobench
  A lightweight ambient occlusion renderer (https://code.google.com/archive/p/aobench)

### asmooth
  Adaptive smoothing (http://www.hcs.harvard.edu/admiralty/)

### asta
  Array of structure of tiled array for data layout transposition (https://github.com/chai-benchmarks/chai)

### atomicIntrinsics
  Atomic add, subtract, min, max, AND, OR, XOR (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### atomicCAS
  64-bit atomic add, min, and max with compare and swap (https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h)

### atomicReduction
  Integer sum reduction with atomics (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/reduction)

### attention
  Ham, T.J., et al., 2020, February. A^ 3: Accelerating Attention Mechanisms in Neural Networks with Approximation. In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA) (pp. 328-341). IEEE.

### axhelm
  Helmholtz matrix-vector product (https://github.com/Nek5000/nekBench/tree/master/axhelm)

### babelstream
  Measure memory transfer rates for copy, add, mul, triad, dot, and nstream (https://github.com/UoB-HPC/BabelStream)

### backprop
  Backpropagation in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bezier-surface
  The Bezier surface (https://github.com/chai-benchmarks/chai)

### bfs
  The breadth-first search in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bh
  Simulate the gravitational forces in a star cluster using the Barnes-Hut n-body algorithm (https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/)

### bilateral
  Bilateral filter (https://github.com/jstraub/cudaPcl)

### binomial
  Evaluate fair call price for a given set of European options under binomial model (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### bitonic-sort
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### bitpacking
  A bit-level operation that aims to reduce the number of bits required to store each value (https://github.com/NVIDIA/nvcomp)

### black-scholes
  The Black-Scholes simulation (https://github.com/cavazos-lab/FinanceBench)

### bm3d
  Block-matching and 3D filtering method for image denoising (https://github.com/DawyD/bm3d-gpu)

### bh
  Simulate the gravitational forces in a star cluster using the Barnes-Hut n-body algorithm (https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/)

### bn
  Bayesian network learning (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks)

### bonds
  Fixed-rate bond with flat forward curve (https://github.com/cavazos-lab/FinanceBench)

### boxfilter
  Box filtering (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### bsearch
  Classic and vectorizable binary search algorithms (https://www.sciencedirect.com/science/article/abs/pii/S0743731517302836)

### bspline-vgh
  Bspline value gradient hessian (https://github.com/QMCPACK/miniqmc/blob/OMP_offload/src/OpenMP/main.cpp)

### bsw
  GPU accelerated Smith-Waterman for performing batch alignments (https://github.com/mgawan/ADEPT)

### burger
  2D Burger's equation (https://github.com/soumyasen1809/OpenMP_C_12_steps_to_Navier_Stokes)

### bwt
  Burrows-Wheeler transform (https://github.com/jedbrooke/cuda_bwt)

### b+tree
  B+Tree in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### ccsd-trpdrv
  The CCSD tengy kernel, which was converted from Fortran to C by Jeff Hammond, in NWChem (https://github.com/jeffhammond/nwchem-ccsd-trpdrv)

### ced
  Canny edge detection (https://github.com/chai-benchmarks/chai)

### cfd
  The CFD solver in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### che
  Phase-field simulation of spinodal decomposition using the Cahn-Hilliard equation (https://github.com/myousefi2016/Cahn-Hilliard-CUDA)

### chemv
  Complex hermitian matrix-vector multiplication (https://repo.or.cz/ppcg.git)

### chi2
  The Chi-square 2-df test. The input data must be downloaded before running the test. Please see README for the link.

### clenergy
  Direct coulomb summation kernel (http://www.ks.uiuc.edu/Training/Workshop/GPU_Aug2010/resources/clenergy.tar.gz)

### clink
  Compact LSTM inference kernel (http://github.com/UCLA-VAST/CLINK)

### cmp
  Seismic processing using the classic common midpoint (CMP) method (https://github.com/hpg-cepetro/IPDPS-CRS-CMP-code)

### cobahh
  Simulation of Random Network of Hodgkin and Huxley Neurons with Exponential Synaptic Conductances (https://dl.acm.org/doi/10.1145/3307339.3343460)

### columnarSolver
  Dimitrov, M. and Esslinger, B., 2021. CUDA Tutorial--Cryptanalysis of Classical Ciphers Using Modern GPUs and CUDA. arXiv preprint arXiv:2103.13937.

### compute-score
  Document filtering (https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/compute-score.html)

### convolutionSeperable 
  Convolution filter of a 2D image with separable kernels (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### crs 
  Cauchy Reed-Solomon encoding (https://www.comp.hkbu.edu.hk/~chxw/gcrs.html)

### d2q9_bgk
  A lattice boltzmann scheme with a 2D grid, 9 velocities, and Bhatnagar-Gross-Krook collision step (https://github.com/WSJHawkins/ExploringSycl)

### dct8x8
  Discrete Cosine Transform (DCT) and inverse DCT for 8x8 blocks (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### degrid
  Radio astronomy degridding (https://github.com/NVIDIA/SKA-gpu-degrid)
  
### deredundancy
  Gene sequence de-redundancy is a precise gene sequence de-redundancy software that supports heterogeneous acceleration (https://github.com/JuZhenCS/gene-sequences-de-redundancy)
  
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

### eigenvalue
  Calculate the eigenvalues of a tridiagonal symmetric matrix (https://github.com/OpenCL/AMD_APP_samples)

### epistatis
  Epistasis detection (https://github.com/rafatcampos/bio-epistasis-detection)
   
### ert
  Modified microkernel in the empirical roofline tool (https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src/master/)
   
### extend2
  Smith-Waterman (SW) extension in Burrow-wheeler aligner for short-read alignment (https://github.com/lh3/bwa)

### extrema
  Find local maxima (https://github.com/rapidsai/cusignal/)

### f16max
  Compute the maximum of half-precision floating-point numbers using bit operations (https://x.momo86.net/en?p=113)

### f16sp
  Half-precision scalar product (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### face
  Face detection using the Viola-Jones algorithm (https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection)

### fdtd3d
  FDTD-3D (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### fhd
  A case study: advanced magnetic resonance imaging reconstruction (https://ict.senecacollege.ca/~gpu610/pages/content/cudas.html)
  
### filter
  Filtering by a predicate (https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

### fft
  FFT in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### flame
  Fractal flame (http://gpugems.hwu-server2.crhc.illinois.edu/)

### floydwarshall
  Floyd-Warshall Pathfinding sample (https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/HIP-Examples-Applications/FloydWarshall/)

### fpc
  Frequent pattern compression ( Base-delta-immediate compression: practical data compression for on-chip caches. In Proceedings of the 21st international conference on Parallel architectures and compilation techniques (pp. 377-
388). ACM.)

### fresnel
  Fresnel integral (http://www.mymathlib.com/functions/fresnel_sin_cos_integrals.html)

### frna
  Accelerate the fill step in predicting the lowest free energy structure and a set of suboptimal structures (http://rna.urmc.rochester.edu/Text/Fold-cuda.html)

### fwt
  Fast Walsh transformation (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### gamma-correction
  Gamma correction (https://github.com/intel/BaseKit-code-samples)

### gaussian
  Gaussian elimination in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### gd
  Gradient descent (https://github.com/CGudapati/BinaryClassification)

### geodesic
  Geodesic distance (https://www.osti.gov/servlets/purl/1576565)

### gmm
  Expectation maximization with Gaussian mixture models (https://github.com/Corv/CUDA-GMM-MultiGPU)

### goulash
  Simulate the dynamics of a small part of a cardiac myocyte, specifically the fast sodium m-gate  (https://github.com/LLNL/goulash)

### grep
  Regular expression matching (https://github.com/bkase/CUDA-grep)

### grrt
  General-relativistic radiative transfer calculations coupled with the calculation of geodesics in the Kerr spacetime (https://github.com/hungyipu/Odyssey)

### haccmk
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

### heartwall
  Heart Wall in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### heat
  The heat equation solver (https://github.com/UoB-HPC/heat_sycl)

### heat2d
  Discrete 2D laplacian operation a number of times on a given vector (https://github.com/gpucw/cuda-lapl)
  
### hellinger
  Hellinger distance (https://github.com/rapidsai/raft)

### henry
  Henry coefficient (https://github.com/CorySimon/HenryCoefficient)

### hexicton
  A Portable and Scalable Solver-Framework for the Hierarchical Equations of Motion (https://github.com/noma/hexciton_benchmark)
  
### histogram
  Histogram (http://github.com/NVlabs/cub/tree/master/experimental)

### hmm
  Hidden markov model (http://developer.download.nvidia.com/compute/DevZone/OpenCL/Projects/oclHiddenMarkovModel.tar.gz)

### hogbom
  The benchmark implements the kernel of the Hogbom Clean deconvolution algorithm (https://github.com/ATNF/askap-benchmarks/)

### hotspot3D
  Hotspot3D in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### hwt1d
  1D Haar wavelet transformation (https://github.com/OpenCL/AMD_APP_samples)

### hybridsort
  Hybridsort in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### interleave
  Interleaved and non-interleaved global memory accesses (Shane Cook. 2012. CUDA Programming: A Developer's Guide to Parallel Computing with GPUs (1st. ed.). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.)

### inversek2j
  The inverse kinematics for 2-joint arm (http://axbench.org/)

### ising
  Monte-Carlo simulations of 2D Ising Model (https://github.com/NVIDIA/ising-gpu/)

### iso2dfd 
  Isotropic 2-dimensional Finite Difference (https://github.com/intel/HPCKit-code-samples/)

### jaccard
  Jaccard index for a sparse matrix (https://github.com/rapidsai/nvgraph/blob/main/cpp/src/jaccard_gpu.cu)

### jacobi
  Jacobi relaxation (https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_gpu/jacobi.cu)

### jenkins-hash 
  Bob Jenkins lookup3 hash function (https://android.googlesource.com/platform/external/jenkins-hash/+/75dbeadebd95869dd623a29b720678c5c5c55630/lookup3.c)

### keccaktreehash 
  A Keccak tree hash function (http://sites.google.com/site/keccaktreegpu/)

### keogh 
  Keogh's lower bound (https://github.com/gravitino/cudadtw)

### kmeans 
  K-means in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### knn
  K-nearest neighbor (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks)

### lanczos
  Lanczos tridiagonalization (https://github.com/linhr/15618)

### langford
  Count planar Langford sequences (https://github.com/boris-dimitrov/z4_planar_langford_multigpu)

### laplace
  A Laplace solver using red-black Gaussian Seidel with SOR solver (https://github.com/kyleniemeyer/laplace_gpu)

### lavaMD
  LavaMD in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### layout
  AoS and SoA comparison (https://github.com/OpenCL/AMD_APP_samples)

### ldpc
  QC-LDPC decoding (https://github.com/robertwgh/cuLDPC)

### lebesgue
  Estimate the Lebesgue constant (https://people.math.sc.edu/Burkardt/c_src/lebesgue/lebesgue.html)

### leukocyte 
  Leukocyte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### libor
  A LIBOR market model Monte Carlo application (https://people.maths.ox.ac.uk/~gilesm/cuda_old.html)

### lid-driven-cavity 
  GPU solver for a 2D lid-driven cavity problem (https://github.com/kyleniemeyer/lid-driven-cavity_gpu)

### linearprobing
  A simple lock-free hash table (https://github.com/nosferalatu/SimpleGPUHashTable)

### lombscargle
  Lomb-Scargle periodogram (https://github.com/rapidsai/cusignal/)

### loopback
  Lookback option simulation (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)

### lsqt
  Linear scaling quantum transport (https://github.com/brucefan1983/gpuqt)

### lud
  LU decomposition in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lulesh
  Livermore unstructured Lagrangian explicit shock hydrodynamics (https://github.com/LLNL/LULESH)

### mandelbrot
  The Mandelbrot set in the HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### marchingCubes
  A practical isosurfacing algorithm for large data on many-core architectures (https://github.com/LRLVEC/MarchingCubes)

### match
  Compute matching scores for two 16K 128D feature points (https://github.com/Celebrandil/CudaSift)

### matrix-rotate
  In-place matrix rotation

### maxpool3d
  3D Maxpooling (https://github.com/nachiket/papaa-opencl)

### md
  Molecular dynamics function in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### mdh
  Simple multiple Debye-Huckel kernel in fast molecular electrostatics algorithms on GPUs (http://gpugems.hwu-server2.crhc.illinois.edu/)
  
### md5hash
  MD5 hash function in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### medianfilter
  Two-dimensional 3x3 median filter of RGBA image (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
### memcpy
  A benchmark for memory copy from a host to a device

### memtest
  Selected memory tests (https://github.com/ComputationalRadiationPhysics/cuda_memtest)

### metropolis
   Simulation of an ensemble of replicas with Metropolis–Hastings computation in the trial run (https://github.com/crinavar/trueke) 

### miniFE
  MiniFE Mantevo mini-application (https://github.com/Mantevo/miniFE)

### minibude
  The core computation of the Bristol University Docking Engine (BUDE) (https://github.com/UoB-HPC/miniBUDE)

### minimap2
  Hardware acceleration of long read pairwise overlapping in genome sequencing (https://github.com/UCLA-VAST/minimap2-acceleration)

### minimod
  A finite difference solver for seismic modeling (https://github.com/rsrice/gpa-minimod-artifacts)

### minisweep
  A deterministic Sn radiation transport miniapp (https://github.com/wdj/minisweep)

### minkowski
  Minkowski distance (https://github.com/rapidsai/raft)

### mis
  Maximal independent set (http://www.cs.txstate.edu/~burtscher/research/ECL-MIS/)

### mixbench
  A read-only version of mixbench (https://github.com/ekondis/mixbench)

### mkl-sgemm
  Single-precision floating-point matrix multiply using Intel<sup>®</sup> Math Kernel Library 

### mmcsf
  MTTKRP kernel using mixed-mode CSF (https://github.com/isratnisa/MM-CSF)

### morphology
  Morphological operators: Erosion and Dilation (https://github.com/yszheda/CUDA-Morphology)

### mt
  Mersenne Twister (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### multimaterial
  Multi-material simulations (https://github.com/reguly/multimaterial)

### murmurhash3
  MurmurHash3 yields a 128-bit hash value (https://github.com/aappleby/smhasher/wiki/MurmurHash3)

### myocte
  Myocte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nbody
  Nbody simulation (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/N-BodyMethods/Nbody)

### nms
  Work-efficient parallel non-maximum suppression kernels (https://github.com/hertasecurity/gpu-nms)

### nn
  Nearest neighbor in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nw
  Needleman-Wunsch in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### page-rank
  PageRank (https://github.com/Sable/Ostrich/tree/master/map-reduce/page-rank)

### particle-diffusion
  Particle diffusion in the HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### particlefilter
  Particle Filter in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### particles
  Particles collision simulation (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### pathfinder
  PathFinder in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### perplexity
  Perplexity search (https://github.com/rapidsai/cuml/)  

### popcount
  Implementations of population count (Jin, Z. and Finkel, H., 2020, May. Population Count on Intel® CPU, GPU and FPGA. In 2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (pp. 432-439). IEEE.)

### pointwise
  Fused point-wise operations (https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/)

### pool
  Pooling layer (https://github.com/PaddlePaddle/Paddle)

### present
  Lightweight cryptography (https://github.com/bozhu/PRESENT-C/blob/master/present.h)

### prna
  Calculate a partition function for a sequence, which can be used to predict base pair probabilities (http://rna.urmc.rochester.edu/Text/partition-cuda.html)

### projectile
  Projectile motion is a program that implements a ballistic equation (https://github.com/intel/BaseKit-code-samples)

### quicksort
  Quicksort (https://software.intel.com/content/www/us/en/develop/download/code-for-the-parallel-universe-article-gpu-quicksort-from-opencl-to-data-parallel-c.html)

### qrg
  Niederreiter quasirandom number generator and Moro's Inverse Cumulative Normal Distribution generator (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### qtclustering
  quality threshold clustering (https://github.com/vetter/shoc/)

### radixsort
  A parallel radix sort (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
### randomAccess
  Random memory access (https://icl.cs.utk.edu/projectsfiles/hpcc/RandomAccess/)
  
### reaction
  3D Gray-Scott reaction diffusion (https://github.com/ifilot/wavefuse)

### reverse
  Reverse an input array of size 256 using shared memory

### rng-wallace
   Random number generation using the Wallace algorithm (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)

### romberg
  Romberg's method (https://github.com/SwayambhuNathRay/Parallel-Romberg-Integration)

### rsbench
  A proxy application for full neutron transport application like OpenMC that support multipole cross section representations
  (https://github.com/ANL-CESAR/RSBench/)

### rtm8
  A structured-grid applications in the oil and gas industry (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/rtm8)

### rushlarsen
  An ODE solver using the Rush-Larsen scheme (https://bitbucket.org/finsberg/gotran/src/master)

### s3d
  Chemical rates computation used in the simulation of combustion (https://github.com/vetter/shoc/)

### scan
  A block-level scan (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

### scan2
  Scan a large array (https://github.com/OpenCL/AMD_APP_samples)

### secp256k1
  Part of BIP39 solver (https://github.com/johncantrell97/bip39-solver-gpu)

### sheath
  Plasma sheath simulation with the particle-in-cell method (https://www.particleincell.com/2016/cuda-pic/)

### shmembench
  The shared local memory microbenchmark (https://github.com/ekondis/gpumembench)

### shuffle
  Shuffle instructions with subgroup sizes of 8, 16, and 32 (https://github.com/cpc/hipcl/tree/master/samples/4_shfl)

### simplemoc
  The attentuation of neutron fluxes across an individual geometrical segment (https://github.com/ANL-CESAR/SimpleMOC-kernel)

### slu
  Sparse LU factorization (https://github.com/sheldonucr/GLU_public)

### snake
  Genome pre-alignment filtering (https://github.com/CMU-SAFARI/SneakySnake)

### sobel
  Sobel filter (https://github.com/OpenCL/AMD_APP_samples)

### sobol
  Sobol quasi-random generator (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### softmax
  The softmax function (https://github.com/pytorch/glow/tree/master/lib/Backends/OpenCL)

### sort
  Radix sort in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### sosfil
  Second-order IIR digital filtering (https://github.com/rapidsai/cusignal/)

### sparkler
  A miniapp for the CoMet comparative genomics application (https://github.com/wdj/sparkler)

### sph
  The simple n^2 SPH simulation (https://github.com/olcf/SPH_Simple)

### split
  The split operation in radix sort (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### spm
  Image registration calculations for the statistical parametric mapping (SPM) system (http://mri.ee.ntust.edu.tw/cuda/)

### sptrsv
  A thread-Level synchronization-free sparse triangular solver (https://github.com/JiyaSu/CapelliniSpTRSV)

### srad
  SRAD (version 1) in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### ss
  String search (https://github.com/OpenCL/AMD_APP_samples)

### sssp
  The single-source shortest path (https://github.com/chai-benchmarks/chai)

### stddev
  Standard deviation (https://github.com/rapidsai/raft)

### stencil1d
  1D stencil (https://www.olcf.ornl.gov/wp-content/uploads/2019/12/02-CUDA-Shared-Memory.pdf)

### stencil3d
  3D stencil (https://github.com/LLNL/cardioid)

### streamcluster
  Streamcluster in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### su3
  Lattice QCD SU(3) matrix-matrix multiply microbenchmark (https://gitlab.com/NERSC/nersc-proxies/su3_bench)

### surfel
   Surfel rendering (https://github.com/jstraub/cudaPcl)

### svd3x3
  Compute the singular value decomposition of 3x3 matrices (https://github.com/kuiwuchn/3x3_SVD_CUDA)

### sw4ck
  SW4 curvilinear kernels are five stencil kernels that account for ~50% of the solution time in SW4 (https://github.com/LLNL/SW4CK)

### testSNAP
  A proxy for the SNAP force calculation in the LAMMPS molecular dynamics package (https://github.com/FitSNAP/TestSNAP)

### thomas
  Solve tridiagonal systems of equations using the Thomas algorithm (https://pm.bsc.es/gitlab/run-math/cuThomasBatch/tree/master)

### tonemapping
  Tone mapping (https://github.com/OpenCL/AMD_APP_samples)

### transpose
  Tensor transposition (https://github.com/Jokeren/GPA-Benchmark/tree/master/ExaTENSOR)

### triad
  Triad in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### tridiagonal
  Matrix solvers for large number of small independent tridiagonal linear systems(http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### tsa
  Trotter-Suzuki approximation (https://bitbucket.org/zzzoom/trottersuzuki/src/master/)

### tsp
  Solving the symmetric traveling salesman problem with iterative hill climbing (https://userweb.cs.txstate.edu/~burtscher/research/TSP_GPU/) 

### urng
  Uniform random noise generator (https://github.com/OpenCL/AMD_APP_samples)
  
### vmc
  Computes expectation values (6D integrals) associated with the helium atom (https://github.com/wadejong/Summer-School-Materials/tree/master/Examples/vmc)

### winograd
  Winograd convolution (https://github.com/ChenyangZhang-cs/iMLBench)

### wyllie
  List ranking with Wyllie's algorithm (Rehman, M. & Kothapalli, Kishore & Narayanan, P.. (2009). Fast and Scalable List Ranking on the GPU. Proceedings of the International Conference on Supercomputing. 235-243. 10.1145/1542275.1542311.)

### xlqc
  Hartree-Fock self-consistent-field (SCF) calculation of H2O (https://github.com/recoli/XLQC) 

### xsbench
  A proxy application for full neutron transport application like OpenMC (https://github.com/ANL-CESAR/XSBench/)


## Developer
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
Bernhard Esslinger, Bert de Jong, Chengjian Liu, Chris Knight, David Oro, Edson Borin, Ian Karlin, Istvan Reguly, Jason Lau, Jeff Hammond, Wayne Joubert, Jiya Su, John Tramm, Ju Zheng, Martin Burtscher, Matthias Noack, Michael Kruse, Michel Migdal, Mike Giles, Mohammed Alser, Muhammad Haseeb, Muaaz Awan, Nevin Liber, Pedro Valero Lara, Piotr Różański, Rahulkumar Gayatri, Shaoyi Peng, Robert Harrison, Thomas Applencourt, Tobias Baumann, Usman Roshan, Ye Luo, Yongbin Gu, Zhe Chen 


Results presented were obtained using the Chameleon testbed supported by the National Science Foundation and the Intel<sup>®</sup> DevCloud. The project also used resources at the Experimental Computing Laboratory (ExCL) at Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.	
