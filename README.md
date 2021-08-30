# oneAPI Direct Programming
This repository contains a collection of data-parallel programs written with CUDA, HIP, SYCL, and OpenMP-4.5 target offloading. Intel<sup>®</sup> DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program in which memory management migration is implemented using the explicit and restricted Unified Shared Memory extension (DPCT usm) or the DPCT header files (DPCT header).

# Experimental Results
Early results are shown [here](results/README.md)


# Reference
### ace
  Phase-field simulation of dendritic solidification (https://github.com/myousefi2016/Allen-Cahn-CUDA)

### aes
  AES encrypt and decrypt (https://github.com/Multi2Sim/m2s-bench-amdsdk-2.5-src)

### affine
  Affine transformation (https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/affine)

### align-types
  Alignment specification for variables of structured types (http://docs.nvidia.com/cuda/cuda-samples/index.html)

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

### atomicCAS
  64-bit atomic add, min, and max with compare and swap (https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h)

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

### binomial
  Evaluate fair call price for a given set of European options under binomial model (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### bitonic-sort
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### black-scholes
  The Black-Scholes simulation (https://github.com/cavazos-lab/FinanceBench)

### bm3d
  Block-matching and 3D filtering method for image denoising (https://github.com/DawyD/bm3d-gpu)

### bh
  Simulate the gravitational forces in a star cluster using the Barnes-Hut n-body algorithm (https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/)

### bn
  Bayesian network learning (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks)

### boxfilter
  Box filtering (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### bsearch
  Classic and vectorizable binary search algorithms (https://www.sciencedirect.com/science/article/abs/pii/S0743731517302836)

### bspline-vgh
  Bspline value gradient hessian (https://github.com/QMCPACK/miniqmc/blob/OMP_offload/src/OpenMP/main.cpp)

### bsw
  GPU accelerated Smith-Waterman for performing batch alignments (https://github.com/mgawan/GPU-BSW)

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

### dtw
  Dynamic time warping for time series (https://github.com/alexkyllo/cuTimeWarp)

### dxtc1
  DXT1 compression (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### easyWave
  Simulation of tsunami generation and propagation in the context of early warning (https://gitext.gfz-potsdam.de/geoperil/easyWave)

### eigenvalue
  Calculate the eigenvalues of a tridiagonal symmetric matrix (https://github.com/OpenCL/AMD_APP_samples)

### epistatis
  Epistasis detection (https://github.com/rafatcampos/bio-epistasis-detection)
   
### extend2
  Smith-Waterman (SW) extension in Burrow-wheeler aligner for short-read alignment (https://github.com/lh3/bwa)

### extrema
  Find local maxima (https://github.com/rapidsai/cusignal/)

### f16sp
  Half-precision scalar product (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### face
  Face detection using the Viola-Jones algorithm (https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection)

### fdtd3d
  FDTD-3D (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

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

### grep
  Regular expression matching (https://github.com/bkase/CUDA-grep)

### haccmk
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

### heartwall
  Heart Wall in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### heat
  The heat equation solver (https://github.com/UoB-HPC/heat_sycl)

### heat2d
  Discrete 2D laplacian operation a number of times on a given vector (https://github.com/gpucw/cuda-lapl)
  
### hexicton
  A Portable and Scalable Solver-Framework for the Hierarchical Equations of Motion (https://github.com/noma/hexciton_benchmark)
  
### histogram
  Histogram (http://github.com/NVlabs/cub/tree/master/experimental)

### hmm
  Hidden markov model (http://developer.download.nvidia.com/compute/DevZone/OpenCL/Projects/oclHiddenMarkovModel.tar.gz)

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

### leukocyte 
  Leukocyte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### libor
  A LIBOR market model Monte Carlo application (https://people.maths.ox.ac.uk/~gilesm/cuda_old.html)

### lid-driven-cavity 
  GPU solver for a 2D lid-driven cavity problem (https://github.com/kyleniemeyer/lid-driven-cavity_gpu)

### lombscargle
  Lomb-Scargle periodogram (https://github.com/rapidsai/cusignal/)

### lsqt
  Linear scaling quantum transport (https://github.com/brucefan1983/gpuqt)

### lud
  LU decomposition in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lulesh
  Livermore unstructured Lagrangian explicit shock hydrodynamics (https://github.com/LLNL/LULESH)

### match
  Compute matching scores for two 16K 128D feature points (https://github.com/Celebrandil/CudaSift)

### matrix-mul
  Single-precision floating-point matrix multiply

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

### prna
  Calculate a partition function for a sequence, which can be used to predict base pair probabilities (http://rna.urmc.rochester.edu/Text/partition-cuda.html)

### projectile
  Projectile motion is a program that implements a ballistic equation (https://github.com/intel/BaseKit-code-samples)

### quicksort
  Quicksort (https://software.intel.com/content/www/us/en/develop/download/code-for-the-parallel-universe-article-gpu-quicksort-from-opencl-to-data-parallel-c.html)

### qrg
  Niederreiter quasirandom number generator and Moro's Inverse Cumulative Normal Distribution generator (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

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

### shmembench
  The shared local memory microbenchmark (https://github.com/ekondis/gpumembench)

### shuffle
  Shuffle instructions with subgroup sizes of 8, 16, and 32 (https://github.com/cpc/hipcl/tree/master/samples/4_shfl)

### simplemoc
  The attentuation of neutron fluxes across an individual geometrical segment (https://github.com/ANL-CESAR/SimpleMOC-kernel)

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

### sptrsv
  A thread-Level synchronization-free sparse triangular solver (https://github.com/JiyaSu/CapelliniSpTRSV)

### srad
  SRAD (version 1) in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### ss
  String search (https://github.com/OpenCL/AMD_APP_samples)

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
  Solve tridiagonal systems of equations using the Thomas algorithm (https://pm.bsc.es/gitlab/run-math/cuThomasBatch/tree/master)

### tonemapping
  Tone mapping (https://github.com/OpenCL/AMD_APP_samples)

### transpose
  Tensor transposition (https://github.com/Jokeren/GPA-Benchmark/tree/master/ExaTENSOR)

### triad
  Triad in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### tridiagonal
  Matrix solvers for large number of small independent tridiagonal linear systems(http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### tsp
  Solving the symmetric traveling salesman problem with iterative hill climbing (https://userweb.cs.txstate.edu/~burtscher/research/TSP_GPU/) 

### urng
  Uniform random noise generator (https://github.com/OpenCL/AMD_APP_samples)
  
### vmc
  Computes expectation values (6D integrals) associated with the helium atom (https://github.com/wadejong/Summer-School-Materials/tree/master/Examples/vmc)

### winograd
  Winograd convolution (https://github.com/ChenyangZhang-cs/iMLBench)

### xsbench
  A proxy application for full neutron transport application like OpenMC
  (https://github.com/ANL-CESAR/XSBench/)


## Developer
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
I would like to thank people who answered my questions about their codes: 
Bernhard Esslinger, Bert de Jong, Chengjian Liu, David Oro, Edson Borin, Ian Karlin, Istvan Reguly, Jason Lau, Jeff Hammond, Wayne Joubert, Jiya Su, John Tramm, Ju Zheng, Matthias Noack, Mike Giles, Mohammed Alser, Muaaz Awan, Nevin Liber, Pedro Valero Lara, Piotr Różański, Rahulkumar Gayatri, Robert Harrison, Thomas Applencourt, Tobias Baumann, Usman Roshan, Ye Luo, Yongbin Gu, Zhe Chen 


Results presented were obtained using the Chameleon testbed supported by the National Science Foundation and the Intel<sup>®</sup> DevCloud. The project also used resources at the Experimental Computing Laboratory (ExCL) at Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.	
