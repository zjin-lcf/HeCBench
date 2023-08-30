# HeCBench
This repository contains a collection of heterogeneous computing benchmarks written with CUDA, HIP, SYCL/DPC++, and OpenMP-4.5 target offloading for studying performance, portability, and productivity. 

## Background, use cases and future work
Z. Jin and J. S. Vetter, "A Benchmark Suite for Improving Performance Portability of the SYCL Programming Model," 2023 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), Raleigh, NC, USA, 2023, pp. 325-327, doi: 10.1109/ISPASS57527.2023.00041. (https://ieeexplore.ieee.org/document/10158214)

# Software installation
[AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)  
[Intel DPC++ compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) or [Intel oneAPI toolkit](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html)  
[NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)

# Dependencies
  Certain SYCL benchmarks require [oneDPL](https://github.com/oneapi-src/oneDPL), [oneTBB](https://github.com/oneapi-src/oneTBB), [Syclomatic](https://github.com/oneapi-src/SYCLomatic), or [oneMKL interfaces](https://github.com/oneapi-src/oneMKL).

# Benchmark categories
Each benchmark falls into a single category. While such classification is not accurate, the arrangement serves as a starting point for users of the benchmark suite. Please see the Reference for more information about each benchmark. 

### Automotive
    daphne

### Bandwidth
    cmembench, babelstream, memcpy, memtest, randomAccess, shmembench, triad 

### Bioinformatics
    all-pairs-distance, bsw, ccs, cm, deredundancy, diamond, epistasis, extend2, frna, fsm, ga, logan, minibude, minimap2, nbnxm, nw, pcc, prna, sa, snake

### Computer vision and image processing
    affine, aobench, asmooth, background-subtract, bezier-surface, bilateral, bm3d, boxfilter, cbsfil, car, ced, colorwheel, convolution1D, convolutionSeperable, dct8x8, debayer, depixel, degrid, doh, dpid, egs, face, flame, gabor, gamma-correction, hogbom, mandelbrot, marchCubes, match, medianfilter, morphology, mriQ, ne, perlin, sobel, tonemapping, recursiveGaussian, resize, sad, seam-carving, spm, srad, ssim, stencil1d, stencil3d, surfel, zoom
    
### Cryptography
    aes, bitcracker, chacha20, columnarSolver, ecdh, keccaktreehash, merkle, present  

### Data compression and reduction
    atomicAggregate, atomicCAS, atomicCost, atomicIntrinsics, atomicPerf, atomicSystemWide, bitpacking, bscan, bwt, compute-score, contract, dxt1, filter, fpc, histogram, minmax, mpc, mtf, rle, sc, scan, scan2, scan3, segment-reduce

### Data encoding, decoding, or verification
    ans, crc64, crs, entropy, jenkins-hash, ldpc, md5hash, murmurhash3

### Finance
    aop, black-scholes, binomial, bonds, libor

### Geoscience
    aidw, coordinates, geodesic, hausdorff, haversine, stsg

### Graph and Tree
    cc, floydwarshall, floydwarshall2, gc, hbc, hungarian, mis, sssp, rsmt

### Language and kernel features
    aligned-types, asta, collision, concurrentKernels, conversion, copy, dispatch, ert, interleave, layout, mallocFree, maxFlops, mixbench, mkl-sgemm, nosync, openmp, overlap, p2p, pad, pitch, popcount, prefetch, reverse, ring, saxpy-ompt, shuffle, simpleMultiDevice, tensorAccessor, threadfence, vote, wordcount, zerocopy 

### Machine learning  
    accuracy, adam, addBiasResidualLayerNorm, attention, attentionMultiHead, backprop, bincount, bn, channelShuffle, channelSum, clink, concat, crossEntropy, dense-embedding, dropout, dwconv, expdist, flip, gd, gelu, ge-spmm, glu, gmm, gru, kalman, kmc, kmeans, knn, lda, lif, logprob, lr, lrn, mask, matern, maxpool3d, mcpr, meanshift, mf-sgd, mmcsf, mnist, mrc, multinomial, nlll, nonzero, overlay, p4, page-rank, perplexity, pointwise, pool, qtclustering, remap, relu, resnet-kernels, rowwiseMoments, sampling, scel, softmax, stddev, streamcluster, swish, unfold, vol2col, wedford, winograd, word2vec

### Math
    atan2, complex, cross, determinant, divergence, dp, eigenvalue, f16max, f16sp, frechet, fresnel, fwt, gaussian, geam, gemmEx, hellinger, hmm, idivide, interval, jaccard, jacobi, kurtosis, lanczos, langford, lci, lebesgue, leukocyte, lfib4, log2, lud, michalewicz, matrix-rotate, matrixT, minkowski, mr, norm2, nqueen, ntt, phmm, pnpoly, rfs, romberg, rsc, secp256k1, simpleSpmv, slu, spmm, sptrsv, thomas, wyllie, zeropoint
   
### Random number generation
    mt, permutate, qrg, rng-wallace, sobol, urng

### Search
    bfs, bsearch, b+tree, grep, keogh, s8n, ss, tsp

### Signal processing
    extrema, fft, lombscargle, sosfil, zmddft

### Simulation
    ace, adv, amgmk, axhelm, bh, bspline-vgh, burger, cooling, ccsd-trpdrv, che, chemv, chi2, clenergy, cmp, cobahh, d2q9_bgk, d3q19_bgk, damage, ddbp, dslash, easyWave, eikonal, fdtd3d, feynman-kac, fhd, fluidSim, gibbs, goulash, gpp, grrt, haccmk, halo-finder, heartwall, heat, heat2d, henry, hexicton, hotspot3D, hwt1d, hypterm, ising, iso2dfd, laplace, laplace3d, lavaMD, lid-driven-cavity, loopback, lsqt, lulesh, mcmd, md, mdh, metropolis, miniFE, minimod, minisweep, miniWeather, multimaterial, myocte, nbody, particle-diffusion, particlefilter, particles, pathfinder, pns, projectile, pso, rainflow, reaction, rsbench, rtm8, rushlarsen, s3d, su3sheath, simplemoc, slit, sparkler, sph, sw4ck, tensorT, testSNAP, tissue, tpacf, tqs, tridiagonal, tsa, vanGenuchten, vmc, wlcpow, wsm5, xlqc, xsbench

### Sorting
    bitonic-sort, hybridsort, is, merge, quicksort, radixsort, segsort, sort, sortKV, split, warpsort

### Robotics
    inversek2j, rodrigues


# Run a benchmark
  Option 1: Makefile scripts that build and run an individual benchmark 
  
      Navigate to a benchmark in CUDA (benchmark-cuda) and type  
      `make ARCH=sm_70 run`  // run on a NIVIDA GPU device with compute capability 7.0
      
      Navigate to a benchmark in HIP (benchmark-hip) and type  
      `make run`
      
      Navigate to a benchmark in SYCL (benchmark-sycl) and type   
     `make CUDA=yes CUDA_ARCH=sm_70 GCC_TOOLCHAIN="" run` (targeting an NVIDIA GPU)
     `make HIP=yes HIP_ARCH=gfx908 run`                   (targeting an AMD GPU)  
     `make run` or `make CC=icpx run`                     (targeting an Intel GPU)
      NOTE: `--gcc-toolchain` may be required for the SYCL compiler to select the proper GNU toolchain; otherwise unset GCC_TOOLCHAIN
     
      Navigate to a benchmark in OpenMP (benchmark-omp) and type  
      `make -f Makefile.nvc run`  (targeting NVIDIA GPUs)
      `make -f Makefile.aomp run` (targeting AMD GPUs)
      `make run`                  (targeting Intel GPUs) 
      
      Users may need to set appropriate values (e.g., `sm_80`, `sm_90`, `gfx906`, `gfx1030`) for their target offloading devices  
      `make -f Makefile.nvc SM=cc80 run`
      `make -f Makefile.aomp ARCH=gfx906 run`
      
  Option 2: Python scripts that help build, run and gather results from the benchmarks. As well as a basic script to compare results from two different runs.

    It works with a `.json` file containing the benchmark names, a regex to
    find the timings in the benchmark output and optional arguments that
    must be provided to the benchmark binary. The `subset.json` contains
    roughly 70 of the benchmarks for cuda, hip and sycl at the moment, more
    work would be required to support the rest of the benchmarks. In
    addition if there are failing benchmarks in the `.json` list, an
    additional text file can be provided with a list of benchmarks to skip
    when running all of them. Benchmarks in the text file can still be run
    explicitly.

    For example to run all the SYCL benchmarks and then all the CUDA
    benchmarks and compare the two:

    ```
    ./autohecbench.py sycl -o sycl.csv
    ./autohecbench.py cuda -o cuda.csv
    ./autohecbench-compare.py sycl.csv cuda.csv
    ```

    It can also be used to run a single benchmark:

    ```
    ./autohecbench.py backprop-sycl --verbose
    ```

    By default it will run a warmup iteration before running each benchmark,
    and it is possible to run the benchmarks multiple times with `-r`:
    ```
    ./autohecbench.py backprop-sycl -r 20 -o mandel.csv
    ```

    And it also has options to pick the SM version or HIP architecture and a
    few other parameters.

# Dataset
For Rodinia benchmarks, please download the dataset at http://lava.cs.virginia.edu/Rodinia/download.htm  
For other benchmarks, datasets are either included with the repository or could be downloaded through the links to the benchmarks 

# Known issues
The programs have not been evaluated on Windows or MacOS  
The lastest Intel SYCL compiler (not the Intel oneAPI toolkit) may be needed for building some SYCL programs successfully  
Kernel results do not exactly match using these programming languages on a platform for certain programs  
Not all programs automate the verification of host and device results  
Not all CUDA programs have SYCL, HIP or OpenMP equivalents  
Not all programs have OpenMP target offloading implementations  
Raw performance of any program may be suboptimal  
Some programs may take long to complete on an integrated GPU  
Some host programs contain platform-specific intrinsics, so they may cause compile error on a PowerPC platform

# Feedback
I appreciate your feedback when any examples don't look right.

# Experimental Results
Early results are shown [here](results/README.md)

# Reference
### accuracy (cuda)
  Accuracy of prediction (https://pytorch.org/)

### ace (cuda)
  Phase-field simulation of dendritic solidification (https://github.com/myousefi2016/Allen-Cahn-CUDA)

### adam (cuda)
  Adaptive moment estimation (https://github.com/hpcaitech/ColossalAI)

### addBiasResidualLayerNorm (cuda)
  Combines the bias, residual of previous block and the computation of layer normalization (https://github.com/NVIDIA/FasterTransformer)

### adv (cuda)
  Advection (https://github.com/Nek5000/nekBench/tree/master/adv)

### aes (opencl)
  AES encrypt and decrypt (https://github.com/Multi2Sim/m2s-bench-amdsdk-2.5-src)

### affine (opencl)
  Affine transformation (https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/affine)

### aidw (cuda)
  Adaptive inverse distance weighting (Mei, G., Xu, N. & Xu, L. Improving GPU-accelerated adaptive IDW interpolation algorithm using fast kNN search. SpringerPlus 5, 1389 (2016))

### aligned-types (cuda)
  Alignment specification for variables of structured types (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### all-pairs-distance (cuda)
  All-pairs distance calculation (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2910913/)

### amgmk (openmp)
  The relax kernel in the AMGmk benchmark (https://asc.llnl.gov/CORAL-benchmarks/Micro/amgmk-v1.0.tar.gz)
  
### ans (cuda)
  Asymmetric numeral systems decoding (https://github.com/weissenberger/multians)
  
### aobench (openmp)
  A lightweight ambient occlusion renderer (https://code.google.com/archive/p/aobench)

### aop (cuda)
  American options pricing (https://github.com/NVIDIA-developer-blog)

### asmooth (cuda)
  Adaptive smoothing (http://www.hcs.harvard.edu/admiralty/)

### asta (cuda)
  Array of structure of tiled array for data layout transposition (https://github.com/chai-benchmarks/chai)

### atan2 (cpp)
  Approximate the atan2 math function (https://github.com/cms-patatrack/pixeltrack-standalone)

### atomicAggreate (cuda)
  Atomic aggregate (https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) 

### atomicIntrinsics (cuda)
  Atomic add, subtract, min, max, AND, OR, XOR (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### atomicCAS (cuda)
  64-bit atomic add, min, and max with compare and swap (https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h)

### atomicCost
  Evaluate the cost of atomic add operations

### atomicPerf (cuda)
  Evaluate atomic add operations over global and shared memory (https://stackoverflow.com/questions/22367238/cuda-atomic-operation-performance-in-different-scenarios)

### atomicReduction (hip)
  Integer sum reduction with atomics (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/reduction)

### atomicSystemWide (cuda)
  System-wide atomics (http://docs.nvidia.com/cuda/cuda-samples/index.html) 

### attention (pseudocodes)
  Ham, T.J., et al., 2020, February. A^ 3: Accelerating Attention Mechanisms in Neural Networks with Approximation. In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA) (pp. 328-341). IEEE.

### attentionMultiHead (cuda)
  Implementation of multi-head attention (https://github.com/IrishCoffee/cudnnMultiHeadAttention)

### axhelm (cuda)
  Helmholtz matrix-vector product (https://github.com/Nek5000/nekBench/tree/master/axhelm)

### babelstream (cuda)
  Measure memory transfer rates for copy, add, mul, triad, dot, and nstream (https://github.com/UoB-HPC/BabelStream)

### background-subtract (cuda)
  Background subtraction (Alptekin Temizel et al. Experiences on Image and Video Processing with CUDA and OpenCL, In Applications of GPU Computing Series, GPU Computing Gems Emerald Edition, Morgan Kaufmann, 2011, Pages 547-567)

### backprop (opencl)
  Backpropagation in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bezier-surface (opencl)
  The Bezier surface (https://github.com/chai-benchmarks/chai)

### bfs (opencl)
  The breadth-first search in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bh (cuda)
  Simulate the gravitational forces in a star cluster using the Barnes-Hut n-body algorithm (https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/)

### bilateral (cuda)
  Bilateral filter (https://github.com/jstraub/cudaPcl)

### bincount (cuda)
  Count the number of values that fall into each bin (https://pytorch.org/)

### binomial (cuda)
  Evaluate fair call price for a given set of European options under binomial model (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### bitcracker (cuda)
  Open-source password cracking tool for storage devices (https://github.com/e-ago/bitcracker.git)

### bitonic-sort (sycl)
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### bitpacking (cuda)
  A bit-level operation that aims to reduce the number of bits required to store each value (https://github.com/NVIDIA/nvcomp)

### black-scholes (cuda)
  The Black-Scholes simulation (https://github.com/cavazos-lab/FinanceBench)

### bm3d (cuda)
  Block-matching and 3D filtering method for image denoising (https://github.com/DawyD/bm3d-gpu)

### bn (cuda)
  Bayesian network learning (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks)

### bonds (cuda)
  Fixed-rate bond with flat forward curve (https://github.com/cavazos-lab/FinanceBench)

### boxfilter (cuda)
  Box filtering (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### bscan (cuda)
  Binary scan in a block (Harris, M. and Garland, M., 2012. Optimizing parallel prefix operations for the Fermi architecture. In GPU Computing Gems Jade Edition (pp. 29-38). Morgan Kaufmann.)

### bsearch (cuda)
  Classic and vectorizable binary search algorithms (https://www.sciencedirect.com/science/article/abs/pii/S0743731517302836)

### bspline-vgh (openmp)
  Bspline value gradient hessian (https://github.com/QMCPACK/miniqmc/blob/OMP_offload/src/OpenMP/main.cpp)

### bsw (cuda)
  GPU accelerated Smith-Waterman for performing batch alignments (https://github.com/mgawan/ADEPT)

### burger (openmp)
  2D Burger's equation (https://github.com/soumyasen1809/OpenMP_C_12_steps_to_Navier_Stokes)

### bwt (cuda)
  Burrows-Wheeler transform (https://github.com/jedbrooke/cuda_bwt)

### b+tree (opencl)
  B+Tree in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### car (cuda)
  Content adaptive resampling (https://github.com/sunwj/CAR)

### cbsfil (cuda)
  Cubic b-spline filtering (https://github.com/DannyRuijters/CubicInterpolationCUDA)

### cc (cuda)
  Connected components (https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/)

### ccs (cuda)
  Condition-dependent Correlation Subgroups (https://github.com/abhatta3/Condition-dependent-Correlation-Subgroups-CCS)

### ccsd-trpdrv (c)
  The CCSD tengy kernel, which was converted from Fortran to C by Jeff Hammond, in NWChem (https://github.com/jeffhammond/nwchem-ccsd-trpdrv)

### ced (opencl)
  Canny edge detection (https://github.com/chai-benchmarks/chai)

### cfd (opencl)
  The CFD solver in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### chacha20 (c)
  ChaCha20 stream cipher (https://github.com/983/ChaCha20)

### channelShuffle (cuda)
  Divide the channels in a tensor into groups and rearrange them (https://pytorch.org/)

### channelSum (cuda)
  Per-channel sum of values (https://pytorch.org/)

### che (cuda)
  Phase-field simulation of spinodal decomposition using the Cahn-Hilliard equation (https://github.com/myousefi2016/Cahn-Hilliard-CUDA)

### chemv (cuda)
  Complex hermitian matrix-vector multiplication (https://repo.or.cz/ppcg.git)

### chi2 (cuda)
  The Chi-square 2-df test. (https://web.njit.edu/~usman/courses/cs677_spring19/)

### clenergy (opencl)
  Direct coulomb summation kernel (http://www.ks.uiuc.edu/Training/Workshop/GPU_Aug2010/resources/clenergy.tar.gz)

### clink (c)
  Compact LSTM inference kernel (http://github.com/UCLA-VAST/CLINK)

### cm (cuda)
  Gene expression connectivity mapping (https://pubmed.ncbi.nlm.nih.gov/24112435/)

### cmembench (cuda)
  The constant memory microbenchmark (https://github.com/ekondis/gpumembench)

### cmp (cuda)
  Seismic processing using the classic common midpoint (CMP) method (https://github.com/hpg-cepetro/IPDPS-CRS-CMP-code)

### cobahh (opencl)
  Simulation of Random Network of Hodgkin and Huxley Neurons with Exponential Synaptic Conductances (https://dl.acm.org/doi/10.1145/3307339.3343460)

### collision (cuda)
  Check collision of duplicate values (https://github.com/facebookarchive/fbcuda)

### colorwheel (c/c++)
   Color encoding of flow vectors (https://vision.middlebury.edu/flow/code/flow-code/colorcode.cpp)

### columnarSolver (cuda)
  Dimitrov, M. and Esslinger, B., 2021. CUDA Tutorial--Cryptanalysis of Classical Ciphers Using Modern GPUs and CUDA. arXiv preprint arXiv:2103.13937.

### complex (cuda)
  Complex numbers arithmetics (https://github.com/tpn/cuda-samples/blob/master/v8.0/include/cuComplex.h)

### compute-score (opencl)
  Document filtering (https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/compute-score.html)

### concat (cuda)
  Concatenation of two tensors (https://github.com/bytedance/lightseq)

### concurrentKernels (cuda)
  Demonstrate the use of streams for concurrent execution of several kernels with dependency on a device (https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/concurrentKernels)

### contract (cuda)
  Second-order tensor aggregation with an adjacency matrix (https://github.com/HyTruongSon/GraphFlow)

### conversion (sycl)
  Conversion among common data types (https://github.com/intel/llvm/issues/7195)

### convolution1D (cuda)
  1D convolution (Kirk, D.B. and Wen-Mei, W.H., 2016. Programming massively parallel processors: a hands-on approach. Morgan kaufmann)

### convolutionSeperable (opencl)
  Convolution filter of a 2D image with separable kernels (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### cooling (cuda)
  Primordial hydrogen/helium cooling curve (https://github.com/cholla-hydro/cholla)

### coordinates (cuda)
  Coordinates(latitude and longitude) transformation using the STL transform (https://github.com/rapidsai/cuspatial)

### copy (cuda)
  Memory copies using direct, zero, and managed memory accesses

### crc64 (openmp)
  64-bit cyclic-redundancy check (https://xgitlab.cels.anl.gov/hfinkel/hpcrc64/-/wikis/home)

### cross (cuda)
  Cross product of two 2D tensors (https://pytorch.org/)

### crossEntropy (sycl)
  Cross entropy loss in the backward phase (https://github.com/intel/llvm/issues/5969)

### crs (cuda)
  Cauchy Reed-Solomon encoding (https://www.comp.hkbu.edu.hk/~chxw/gcrs.html)

### crc64 (openmp)
  64-bit cyclic-redundancy check (https://xgitlab.cels.anl.gov/hfinkel/hpcrc64/-/wikis/home)

### crs (cuda)
  Cauchy Reed-Solomon encoding (https://www.comp.hkbu.edu.hk/~chxw/gcrs.html)

### d2q9_bgk (sycl)
  A lattice boltzmann scheme with a 2D grid, 9 velocities, and Bhatnagar-Gross-Krook collision step (https://github.com/WSJHawkins/ExploringSycl)

### d3q19_bgk (cuda)
  Lattice Boltzmann simulation framework based on C++ parallel algorithms (https://gitlab.com/unigehpfs/stlbm)

### daphne (cuda)
  The Darmstadt automotive parallel heterogeneous benchmark suite (https://github.com/esa-tu-darmstadt/daphne-benchmark)

### damage (opencl)
  The continuum level damage in a peridynamic body (https://github.com/alan-turing-institute/PeriPy)

### dct8x8 (opencl)
  Discrete Cosine Transform (DCT) and inverse DCT for 8x8 blocks (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### ddbp (cuda)
  Distance-driven backprojection (https://github.com/LAVI-USP/DBT-Reconstruction)

### debayer (opencl)
  Convert a Bayer mosaic raw image to RGB (https://github.com/GrokImageCompression/latke)

### degrid (cuda)
  Radio astronomy degridding (https://github.com/NVIDIA/SKA-gpu-degrid)
  
### dense-embedding (cuda)
  Dense embedding add operations (https://pytorch.org/)
  
### depixel (cuda)
  Check connectivity and remove crosses in depixelization of pixel art (https://github.com/yzhwang/depixelization) 
  
### deredundancy (sycl)
  Gene sequence de-redundancy is a precise gene sequence de-redundancy software that supports heterogeneous acceleration (https://github.com/JuZhenCS/gene-sequences-de-redundancy)

### determinant (cuda)
  Calculate the determinant of a matrix using library-based decomposition and strided reduction (https://github.com/OrangeOwlSolutions/Linear-Algebra)
  
### diamond (opencl)
  Mask sequences kernel in Diamond (https://github.com/bbuchfink/diamond)

### dispatch (hip)
  Kernel dispatch rate and latency (https://github.com/ROCm-Developer-Tools/HIP-CPU)

### distort (cuda)
  Barrel distortion (https://github.com/Cuda-Chen/barrel-distortion-cuda)

### divergence (cuda)
  CPU and GPU divergence test (https://github.com/E3SM-Project/divergence_cmdvse)

### doh (cuda)
  Determinant of a Hessian matrix (https://github.com/rapidsai/cucim)

### dp (opencl)
  Dot product (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### dpid (cuda)
  Detail-preserving image downscaling (https://github.com/mergian/dpid)

### dropout (cuda)
  Randomly zero some elements of the input array with a probability using samples from a uniform distribution (https://github.com/pytorch/)

### dslash (sycl)
  A Lattice QCD Dslash operator proxy application derived from MILC (https://gitlab.com/NERSC/nersc-proxies/milc-dslash)

### dxtc1 (opencl)
  DXT1 compression (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### dwconv (cuda)
  Depth-wise convolution (https://pytorch.org/)

### easyWave (cuda)
  Simulation of tsunami generation and propagation in the context of early warning (https://gitext.gfz-potsdam.de/geoperil/easyWave)

### ecdh (c++)
  Elliptic curve Diffie-Hellman key exchange (https://github.com/jaw566/ECDH)

### egs (cuda)
  Parallel implementation of EGSnrc's photon transport mechanism (https://jonaslippuner.com/research/cuda-egs/)

### eigenvalue (opencl)
  Calculate the eigenvalues of a tridiagonal symmetric matrix (https://github.com/OpenCL/AMD_APP_samples)

### eikonal (cuda)
  Fast iterative method for Eikonal equations on structured volumes (https://github.com/SCIInstitute/StructuredEikonal)

### entropy (cuda)
  Compute the entropy for each point of a 2D matrix using a 5x5 window (https://lan-jing.github.io/parallel%20computing/system/entropy/)

### epistasis (sycl)
  Epistasis detection (https://github.com/rafatcampos/bio-epistasis-detection)
   
### ert (cuda)
  Modified microkernel in the empirical roofline tool (https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src/master/)
   
### expdist (cuda)
  Compute the Bhattacharya cost function (https://github.com/benvanwerkhoven/kernel_tuner)

### extend2 (c)
  Smith-Waterman (SW) extension in Burrow-wheeler aligner for short-read alignment (https://github.com/lh3/bwa)

### extrema (cuda)
  Find local maxima (https://github.com/rapidsai/cusignal/)

### f16max (c)
  Compute the maximum of half-precision floating-point numbers using bit operations (https://x.momo86.net/en?p=113)

### f16sp (cuda)
  Half-precision scalar product (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### face (cuda)
  Face detection using the Viola-Jones algorithm (https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection)

### fdtd3d (opencl)
  FDTD-3D (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### feynman-kac (c)
  Use of Feynman-Kac algorithm to solve Poisson's equation in a 2D ellipse (https://people.sc.fsu.edu/~jburkardt/c_src/feynman_kac_2d/feynman_kac_2d.html)

### fhd (cuda)
  A case study: advanced magnetic resonance imaging reconstruction (https://ict.senecacollege.ca/~gpu610/pages/content/cudas.html)
  
### filter (cuda)
  Filtering by a predicate (https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

### fft (opencl)
  FFT in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### flame (cuda)
  Fractal flame (http://gpugems.hwu-server2.crhc.illinois.edu/)

### flip (cuda)
  Tensor flip (https://pytorch.org/)

### floydwarshall (hip)
  Floyd-Warshall Pathfinding sample (https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/HIP-Examples-Applications/FloydWarshall/)

### floydwarshall2 (cuda)
  Fast Floyd-Warshall for all-pairs-shortest paths (https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/)

### fluidSim (opencl)
  2D Fluid Simulation using the Lattice-Boltzman method (https://github.com/OpenCL/AMD_APP_samples)

### fpc (opencl)
  Frequent pattern compression ( Base-delta-immediate compression: practical data compression for on-chip caches. In Proceedings of the 21st international conference on Parallel architectures and compilation techniques (pp. 377- 388). ACM.)

### frechet (matlab)
  Compute the discrete Frechet distance between two curves specified by discrete ordered points in n-dimensional space (https://github.com/mp4096/discrete-frechet-distance)

### fresnel (c)
  Fresnel integral (http://www.mymathlib.com/functions/fresnel_sin_cos_integrals.html)

### frna (cuda)
  Accelerate the fill step in predicting the lowest free energy structure and a set of suboptimal structures (http://rna.urmc.rochester.edu/Text/Fold-cuda.html)

### fsm (cuda)
  A GPU-accelerated implementation of a genetic algorithm for finding well-performing finite-state machines for predicting binary sequences (https://userweb.cs.txstate.edu/~burtscher/research/FSM_GA/)

### fwt (cuda)
  Fast Walsh transformation (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### ga (cuda)
  Gene alignment (https://github.com/NUCAR-DEV/Hetero-Mark)

### gabor (c/c++)
  Gabor filter function (https://github.com/fercer/gaborfilter)

### gamma-correction (sycl)
  Gamma correction (https://github.com/intel/BaseKit-code-samples)

### gaussian (opencl)
  Gaussian elimination in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### gc (cuda)
  Graph coloring via shortcutting (https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/)

### gd (c++)
  Gradient descent (https://github.com/CGudapati/BinaryClassification)

### geam (cuda)
  Matrix transpose using the BLAS-extension functions (https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam)

### gelu (cuda)
  Apply the Gaussian error linear units function (https://github.com/NVIDIA/FasterTransformer)

### geodesic (opencl)
  Geodesic distance (https://www.osti.gov/servlets/purl/1576565)

### ge-spmm (cuda)
  General-purposed sparse matrix-matrix multiplication on GPUs for graph neural networks (https://github.com/hgyhungry/ge-spmm)

### gemmEx (cuda)
  General matrix-matrix multiplication on GPUs (https://godweiyang.com/2021/08/24/gemm/)

### gibbs (cuda)
  Implementation of a Gibbs-Metropolis sampling algorithm (https://github.com/arendsee/cuda-gibbs-example)

### glu (cuda)
  The gated linear unit function (https://pytorch.org/docs/stable/generated/torch.nn.GLU.html)

### gmm (cuda)
  Expectation maximization with Gaussian mixture models (https://github.com/Corv/CUDA-GMM-MultiGPU)

### goulash (cuda)
  Simulate the dynamics of a small part of a cardiac myocyte, specifically the fast sodium m-gate  (https://github.com/LLNL/goulash)

### gpp (cuda, omp)
  General Plasman Pole Self-Energy Simulation the BerkeleyGW software package (https://github.com/NERSC/gpu-for-science-day-july-2019)

### grep (cuda)
  Regular expression matching (https://github.com/bkase/CUDA-grep)

### grrt (cuda)
  General-relativistic radiative transfer calculations coupled with the calculation of geodesics in the Kerr spacetime (https://github.com/hungyipu/Odyssey)

### gru (cuda)
  Forward operations of a gated recurrent unit (https://pytorch.org/)

### haccmk (c)
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

### halo-finder (cuda)
  Parallel halo finder operation (https://gem5.googlesource.com/public/gem5-resources)

### hausdorff (c/c++)
  Hausdorff distance  (https://github.com/arohamirai/Hausdorff-Distance-Match)

### haversine (cuda)
  Haversine distance  (https://github.com/rapidsai/cuspatial)

### hbc (cuda)
  Hybrid methods for parallel betweenness centrality (https://github.com/Adam27X/hybrid_BC)

### heartwall (opencl)
  Heart Wall in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### heat (sycl)
  The heat equation solver (https://github.com/UoB-HPC/heat_sycl)

### heat2d (cuda)
  Discrete 2D laplacian operation a number of times on a given vector (https://github.com/gpucw/cuda-lapl)
  
### hellinger (cuda)
  Hellinger distance (https://github.com/rapidsai/raft)

### henry (cuda)
  Henry coefficient (https://github.com/CorySimon/HenryCoefficient)

### hexicton (opencl)
  A Portable and Scalable Solver-Framework for the Hierarchical Equations of Motion (https://github.com/noma/hexciton_benchmark)
  
### histogram (cuda)
  Histogram (http://github.com/NVlabs/cub/tree/master/experimental)

### hmm (opencl)
  Hidden markov model (http://developer.download.nvidia.com/compute/DevZone/OpenCL/Projects/oclHiddenMarkovModel.tar.gz)

### hogbom (cuda)
  The benchmark implements the kernel of the Hogbom Clean deconvolution algorithm (https://github.com/ATNF/askap-benchmarks/)

### hotspot3D (opencl)
  Hotspot3D in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### hungarian (cuda)
  Fast block distributed Implementation of the Hungarian Algorithm (https://github.com/paclopes/HungarianGPU)

### hwt1d (opencl)
  1D Haar wavelet transformation (https://github.com/OpenCL/AMD_APP_samples)

### hybridsort (opencl)
  Hybridsort in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### hypterm (cuda)
  A routine from the ExpCNS Compressible Navier-Stokes mini-application (https://github.com/pssrawat/ppopp-artifact)

### idivide (cuda)
  Fast interger divide (https://github.com/milakov/int_fastdiv)

### interleave (cuda)
  Interleaved and non-interleaved global memory accesses (Shane Cook. 2012. CUDA Programming: A Developer's Guide to Parallel Computing with GPUs (1st. ed.). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.)

### interval (cuda)
  Interval arithmetic operators example (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### inversek2j (cuda)
  The inverse kinematics for 2-joint arm (http://axbench.org/)

### is (cuda)
  Integer sort (https://github.com/GMAP/NPB-GPU)

### ising (cuda)
  Monte-Carlo simulations of 2D Ising Model (https://github.com/NVIDIA/ising-gpu/)

### iso2dfd (sycl)
  Isotropic 2-dimensional Finite Difference (https://github.com/intel/HPCKit-code-samples/)

### jaccard (cuda)
  Jaccard index for a sparse matrix (https://github.com/rapidsai/nvgraph/blob/main/cpp/src/jaccard_gpu.cu)

### jacobi (cuda)
  Jacobi relaxation (https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_gpu/jacobi.cu)

### jenkins-hash (c)
  Bob Jenkins lookup3 hash function (https://android.googlesource.com/platform/external/jenkins-hash/+/75dbeadebd95869dd623a29b720678c5c5c55630/lookup3.c)

### kalman (cuda)
  Kalman filter (https://github.com/rapidsai/cuml/)  

### keccaktreehash (cuda)
  A Keccak tree hash function (http://sites.google.com/site/keccaktreegpu/)

### keogh (cuda)
  Keogh's lower bound (https://github.com/gravitino/cudadtw)

### kmc (cuda)
  Kernel matrix compute (https://github.com/MKLab-ITI/CUDA)

### kmeans (opencl)
  K-means in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### knn (cuda)
  K-nearest neighbor (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks)

### kurtosis (cuda)
  Compute the kurtosis of two variables (https://github.com/d-d-j/ddj_store)

### lanczos (cuda)
  Lanczos tridiagonalization (https://github.com/linhr/15618)

### langford (cuda)
  Count planar Langford sequences (https://github.com/boris-dimitrov/z4_planar_langford_multigpu)

### laplace (cuda)
  A Laplace solver using red-black Gaussian Seidel with SOR solver (https://github.com/kyleniemeyer/laplace_gpu)

### laplace3d (cuda)
  Solve Laplace equation on a regular 3D grid (https://github.com/gpgpu-sim/ispass2009-benchmarks)

### lavaMD (opencl)
  LavaMD in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### layout (opencl)
  AoS and SoA comparison (https://github.com/OpenCL/AMD_APP_samples)

### lci (c)
  Landau collisional integral (https://github.com/vskokov/Landau_Collisional_Integral)

### lda (cuda)
  Latent Dirichlet allocation (https://github.com/js1010/cusim)

### ldpc (cuda)
  QC-LDPC decoding (https://github.com/robertwgh/cuLDPC)

### lebesgue (c)
  Estimate the Lebesgue constant (https://people.math.sc.edu/Burkardt/c_src/lebesgue/lebesgue.html)

### leukocyte  (opencl)
  Leukocyte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lfib4 (cuda)
  Marsa-LFIB4 pseudorandom number generator (https://bitbucket.org/przemstp/gpu-marsa-lfib4/src/master/)

### libor (cuda)
  A LIBOR market model Monte Carlo application (https://people.maths.ox.ac.uk/~gilesm/cuda_old.html)

### lid-driven-cavity  (cuda)
  GPU solver for a 2D lid-driven cavity problem (https://github.com/kyleniemeyer/lid-driven-cavity_gpu)

### lif (cuda)
   A leaky integrate-and-fire neuron model (https://github.com/e2crawfo/hrr-scaling)

### linearprobing (cuda)
  A simple lock-free hash table (https://github.com/nosferalatu/SimpleGPUHashTable)

### log2 (c)
  Approximate the log2 math function (https://adacenter.org/sites/default/files/milspec/Transcendentals.zip)

### logan (cuda)
  GPU-based X-Drop alignment (https://github.com/albertozeni/LOGAN)

### logprob (cuda)
  Convert logits to probabilities (https://github.com/NVIDIA/FasterTransformer)

### lombscargle (cuda)
  Lomb-Scargle periodogram (https://github.com/rapidsai/cusignal/)

### loopback (cuda)
  Lookback option simulation (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)

### lr (opencl)
  Linear regression (https://github.com/ChenyangZhang-cs/iMLBench)

### lrn (sycl)
  Local response normalization (https://github.com/intel/llvm/issues/8292)

### lsqt (cuda)
  Linear scaling quantum transport (https://github.com/brucefan1983/gpuqt)

### lud (opencl)
  LU decomposition in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lulesh (cuda)
  Livermore unstructured Lagrangian explicit shock hydrodynamics (https://github.com/LLNL/LULESH)

### mallocFree (hip)
  Memory allocation and deallocation samples (https://github.com/ROCm-Developer-Tools/HIP/)

### mandelbrot (sycl)
  The Mandelbrot set in the HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### marchingCubes (cuda)
  A practical isosurfacing algorithm for large data on many-core architectures (https://github.com/LRLVEC/MarchingCubes)

### mask (cuda)
  Masking operators in Pytorch (https://pytorch.org/)

### match (cuda)
  Compute matching scores for two 16K 128D feature points (https://github.com/Celebrandil/CudaSift)

### matern (cuda)
  Sum using the Matern kernel (https://tbetcke.github.io/hpc_lecture_notes/rbf_evaluation.html)

### matrix-rotate (openmp)
  In-place matrix rotation

### matrixT (cuda)
  Matrix transposition (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### maxpool3d (opencl)
  3D Maxpooling (https://github.com/nachiket/papaa-opencl)

### maxFlops (opencl)
  Maximum floating-point operations in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### mcmd (cuda)
  Monte Carlo and Molecular Dynamics Simulation Package (https://github.com/khavernathy/mcmd)

### mcpr (cuda)
  Multi-category probit regression (https://github.com/berkeley-scf/gpu-workshop-2016)

### md (opencl)
  Molecular dynamics function in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### mdh (opencl)
  Simple multiple Debye-Huckel kernel in fast molecular electrostatics algorithms on GPUs (http://gpugems.hwu-server2.crhc.illinois.edu/)
  
### md5hash (opencl)
  MD5 hash function in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### meanshift (cuda)
  Mean shift clustering (https://github.com/w00zie/mean_shift)
  
### medianfilter (opencl)
  Two-dimensional 3x3 median filter of RGBA image (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
### merkle (sycl)
  Merkle tree construction using rescue prime hash (https://github.com/itzmeanjan/ff-gpu)
  
### memcpy (cuda)
  A benchmark for memory copy from a host to a device

### memtest (cuda)
  Selected memory tests (https://github.com/ComputationalRadiationPhysics/cuda_memtest)

### merge (cuda)
  Merge two unsorted arrays into a sorted array (https://github.com/ogreen/MergePathGPU)

### metropolis (cuda)
   Simulation of an ensemble of replicas with Metropolis–Hastings computation in the trial run (https://github.com/crinavar/trueke) 

### mf-sgd (cuda)
   Matrix factorization with stochastic gradient descent (https://github.com/cuMF/cumf_sgd)

### michalewicz (c)
   Evaluate the Michalewicz function (https://www.sfu.ca/~ssurjano/michal.html)

### miniFE (omp)
  MiniFE Mantevo mini-application (https://github.com/Mantevo/miniFE)

### minibude (sycl)
  The core computation of the Bristol University Docking Engine (BUDE) (https://github.com/UoB-HPC/miniBUDE)

### minimap2 (cuda)
  Hardware acceleration of long read pairwise overlapping in genome sequencing (https://github.com/UCLA-VAST/minimap2-acceleration)

### minimod (cuda)
  A finite difference solver for seismic modeling (https://github.com/rsrice/gpa-minimod-artifacts)

### minisweep (openmp)
  A deterministic Sn radiation transport miniapp (https://github.com/wdj/minisweep)

### miniWeather (openmp)
  A parallel programming training mini-app simulating weather-like flows (https://github.com/mrnorman/miniWeather)

### minkowski (cuda)
  Minkowski distance (https://github.com/rapidsai/raft)

### minmax (cuda)
  Find the smallest and largest elements (https://github.com/rapidsai/cuspatial)

### mis (cuda)
  Maximal independent set (http://www.cs.txstate.edu/~burtscher/research/ECL-MIS/)

### mixbench (cuda)
  A read-only version of mixbench (https://github.com/ekondis/mixbench)

### mkl-sgemm (sycl) 
  Single-precision floating-point matrix multiply using Intel<sup>®</sup> Math Kernel Library 

### mmcsf (cuda)
  MTTKRP kernel using mixed-mode CSF (https://github.com/isratnisa/MM-CSF)

### mnist (cuda)
  Chapter 4.2: Converting CUDA CNN to HIP (https://developer.amd.com/wp-content/resources)

### morphology (cuda)
  Morphological operators: Erosion and Dilation (https://github.com/yszheda/CUDA-Morphology)

### mr (c)
  The Miller-Rabin primality test (https://github.com/wizykowski/miller-rabin)

### mriQ (cuda)
  Computation of a matrix Q used in a 3D magnetic resonance image reconstruction (https://github.com/abduld/Parboil/blob/master/benchmarks/mri-q/DESCRIPTION)

### mrc (cuda)
  Margin ranking criterion operation (https://pytorch.org)

### mt (opencl)
  Mersenne Twister (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### mtf (thrust)
  Move-to-front transform (https://github.com/bzip2-cuda/bzip2-cuda)

### multimaterial (sycl)
  Multi-material simulations (https://github.com/reguly/multimaterial)

### multinomial (cuda)
  Multinomial sampling (https://pytorch.org)

### murmurhash3 (c)
  MurmurHash3 yields a 128-bit hash value (https://github.com/aappleby/smhasher/wiki/MurmurHash3)

### myocte (opencl)
  Myocte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nbnxm (sycl)
  Computing non-bonded pair interactions (https://manual.gromacs.org/current/doxygen/html-full/page_nbnxm.xhtml)

### nbody (opencl)
  Nbody simulation (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/N-BodyMethods/Nbody)

### ne (cuda)
  Normal estimation in 3D (https://github.com/PointCloudLibrary/pcl)

### nlll (cuda)
  The negative log likelihood 2D loss reduction (https://pytorch.org/)

### nms (cuda)
  Work-efficient parallel non-maximum suppression kernels (https://github.com/hertasecurity/gpu-nms)

### nn (opencl)
  Nearest neighbor in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nonzero (cuda)
  Return a tensor containing the indices of all non-zero elements of input (https://pytorch.org/)

### norm2 (cuda)
  Compute the Euclidean norm of a vector (https://docs.nvidia.com/cuda/cublas)

### nosync (cuda)
  Stream synchronization in Thrust and oneDPL (https://github.com/NVIDIA/thrust/tree/main/examples)

### nqueen (cuda)
  N-Queens (https://github.com/tcarneirop/ChOp)

### ntt (cuda)
  Number-theoretic transform (https://github.com/vernamlab/cuHE)

### nw (opencl)
  Needleman-Wunsch in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### openmp (cuda)
  Multi-threading over a single device (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### overlap (cuda)
  Overlap data copies with compute kernels (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### overlay (cuda)
  Overlay grid in the DetectNet (https://github.com/dusty-nv/jetson-inference)

### p2p (cuda)
  Simple peer-to-peer accesses (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### p4 (cuda)
  PointPillar post-processing (https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)

### pad (cuda)
  In-place padding (https://github.com/chai-benchmarks/chai)

### page-rank (opencl)
  PageRank (https://github.com/Sable/Ostrich/tree/master/map-reduce/page-rank)

### particle-diffusion (sycl)
  Particle diffusion in the HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### particlefilter (opencl)
  Particle Filter in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### particles (opencl)
  Particles collision simulation (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### pathfinder (opencl)
  PathFinder in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### pcc (cuda)
  Compute pairwise Pearson’s correlation coefficient (https://github.com/pcdslab/Fast-GPU-PCC)

### perlin (cuda)
  Perlin noise generator (https://github.com/silverweed/perlin_cuda)

### permutate (cuda)
  Parallel implementation of the permutation testing in NIST SP 800-90B (https://github.com/yeah1kim/yeah_GPU_SP800_90B_IID)

### perplexity (cuda)
  Perplexity search (https://github.com/rapidsai/cuml/)  

### phmm (cuda)
  Pair hidden Markov model (https://github.com/lienliang/Pair_HMM_forward_GPU)

### pitch (cuda)
  Pitched memory allocation (https://docs.nvidia.com/cuda/cuda-c-programming-guide)

### pnpoly (cuda)
  Solve the point-in-polygon problem using the crossing number algorithm (https://github.com/benvanwerkhoven/kernel_tuner)

### pns (cuda)
  Petri-net simulation (https://github.com/abduld/Parboil/)

### pointwise (cuda)
  Fused point-wise operations (https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/)

### pool (hip)
  Pooling layer (https://github.com/PaddlePaddle/Paddle)

### popcount (opencl)
  Implementations of population count (Jin, Z. and Finkel, H., 2020, May. Population Count on Intel® CPU, GPU and FPGA. In 2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (pp. 432-439). IEEE.)

### prefetch (hip)
  Concurrent managed accesses (https://github.com/ROCm-Developer-Tools/HIP/)

### present (c)
  Lightweight cryptography (https://github.com/bozhu/PRESENT-C/blob/master/present.h)

### prna (cuda)
  Calculate a partition function for a sequence, which can be used to predict base pair probabilities (http://rna.urmc.rochester.edu/Text/partition-cuda.html)

### projectile (sycl)
  Projectile motion is a program that implements a ballistic equation (https://github.com/intel/BaseKit-code-samples)

### pso (cuda)
  A modified implementation of particle swarm optimization using Levy function (https://github.com/wiseodd/cuda-pso, https://github.com/chensohg/GPU_CUDA_PSO)

### qrg (cuda)
  Niederreiter quasirandom number generator and Moro's Inverse Cumulative Normal Distribution generator (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### qtclustering (opencl)
  quality threshold clustering (https://github.com/vetter/shoc/)

### quicksort (sycl)
  Quicksort (https://software.intel.com/content/www/us/en/develop/download/code-for-the-parallel-universe-article-gpu-quicksort-from-opencl-to-data-parallel-c.html)

### radixsort (opencl)
  A parallel radix sort (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
### radixsort2 (cuda)
  A library-based sort-by-key (https://github.com/NVIDIA/cuda-samples)
  
### rainflow (c)
  A fast rainflow cycle counting algorithm (https://github.com/carlos-souto/rainflow-cycle-counting)
  
### randomAccess (openmp)
  Random memory access (https://icl.cs.utk.edu/projectsfiles/hpcc/RandomAccess/)
  
### reaction (cuda)
  3D Gray-Scott reaction diffusion (https://github.com/ifilot/wavefuse)

### recursiveGaussian (opencl)
  2-dimensional Gaussian Blur Filter of RGBA image (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### remap (cuda)
  Map unique values to indices (https://pytorch.org/)

### relu (cuda)
  Rectified linear unit (https://github.com/tensorflow)

### resize (cuda)
  Resize images (https://github.com/opencv/)

### resnet-kernels (cuda)
  ResNet kernels for inference (https://github.com/xuqiantong/CUDA-Winograd)

### reverse (cuda)
  Reverse an input array of size 256 using shared memory

### rfs (cuda)
  Reproducible floating sum (https://github.com/facebookarchive/fbcuda)

### ring (sycl)
  Non-P2P transfers in a circular manner among GPU devices

### rle (cuda)
  Computes a run-length encoding of a sequence (https://github.com/NVIDIA/cub)

### rng-wallace (cuda)
  Random number generation using the Wallace algorithm (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)

### rodrigues (cuda)
  Rodrigues' rotation (https://github.com/DIDSR/VICTRE_MCGPU)

### romberg (cuda)
  Romberg's method (https://github.com/SwayambhuNathRay/Parallel-Romberg-Integration)

### rowwiseMoments (cuda)
  Compute row-wise moments (https://pytorch.org/)

### rsbench (opencl)
  A proxy application for full neutron transport application like OpenMC that support multipole cross section representations
  (https://github.com/ANL-CESAR/RSBench/)

### rsc (cuda)
  Random sample consensus based on task partitioning (https://github.com/chai-benchmarks/chai)

### rsmt (cuda)
  Rectilinear Steiner minimum tree (https://userweb.cs.txstate.edu/~burtscher/research/SFP/)

### rtm8 (hip)
  A structured-grid applications in the oil and gas industry (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/rtm8)

### rushlarsen (cuda)
  An ODE solver using the Rush-Larsen scheme (https://bitbucket.org/finsberg/gotran/src/master)

### s3d (opencl)
  Chemical rates computation used in the simulation of combustion (https://github.com/vetter/shoc/)

### s8n (cuda)
  Stacked 8-neighborhood search finds nearest neighbors in each of the eight octants partitioned by ordering of three coordinates (https://github.com/MVIG-SJTU/pointSIFT/tree/master)

### sa (cuda)
  Dynamic parallel skew algorithm for suffix array on GPU (https://github.com/gmzang/Parallel-Suffix-Array-on-GPU)

### sad (cuda)
  Naive template matching with SAD (https://github.com/gholomia/CTMC)

### sampling (cuda)
  Shapley sampling values explanation method (https://github.com/rapidsai/cuml)

### saxpy-ompt (openmp)
  Perform the SAXPY operation on host and device (https://github.com/pc2/OMP-Offloading)

### sc (cuda)
  Stream compaction (https://github.com/chai-benchmarks/chai)

### scan (cuda)
  Scan with bank-conflict-aware optimization (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

### scan2 (opencl)
  Scan a large array (https://github.com/OpenCL/AMD_APP_samples)

### scan3 (cuda)
  Scan a large array using vendors' library (https://github.com/OpenCL/AMD_APP_samples)

### scel (cuda)
  Sigmoid cross-entropy with logits (https://pytorch.org/)

### score (cuda)
  Find the top scores (https://github.com/opencv/)

### secp256k1 (cuda)
  Part of BIP39 solver (https://github.com/johncantrell97/bip39-solver-gpu)

### seam-carving (cuda)
  Seam carving (https://github.com/pauty/CUDA_seam_carving)

### segment-reduce (cuda)
  Segmented reduction using Thrust and oneDPL (https://github.com/c3sr/tcu_scope)

### segsort (cuda)
  Fast segmented sort on a GPU (https://github.com/Funatiq/bb_segsort)

### sheath (cuda)
  Plasma sheath simulation with the particle-in-cell method (https://www.particleincell.com/2016/cuda-pic/)

### shmembench (cuda)
  The shared local memory microbenchmark (https://github.com/ekondis/gpumembench)

### shuffle (hip)
  Shuffle instructions with subgroup sizes of 8, 16, and 32 (https://github.com/cpc/hipcl/tree/master/samples/4_shfl)

### si (cuda)
  Set intersection with matrix multiply (https://github.com/chribell/set_intersection)

### simplemoc (opencl)
  The attentuation of neutron fluxes across an individual geometrical segment (https://github.com/ANL-CESAR/SimpleMOC-kernel)

### simpleMultiDevice (cuda)
  Execute kernels on multiple devices (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### simpleSpmv (cuda)
  Simple sparse matrix vector multiply (https://github.com/passlab/CUDAMicroBench) 

### slit (cuda)
  Slit experiment to compute diffraction patterns (https://github.com/bamaratunga/cuda_fft.git)

### slu (cuda)
  Sparse LU factorization (https://github.com/sheldonucr/GLU_public)

### snake (cuda)
  Genome pre-alignment filtering (https://github.com/CMU-SAFARI/SneakySnake)

### sobel (opencl)
  Sobel filter (https://github.com/OpenCL/AMD_APP_samples)

### sobol (cuda)
  Sobol quasi-random generator (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### softmax (opencl)
  The softmax function (https://github.com/pytorch/glow/tree/master/lib/Backends/OpenCL)

### sort (opencl)
  Radix sort in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### sortKV (cuda)
  Sort by key using Thrust and oneDPL

### sosfil (cuda)
  Second-order IIR digital filtering (https://github.com/rapidsai/cusignal/)

### sparkler (cuda)
  A miniapp for the CoMet comparative genomics application (https://github.com/wdj/sparkler)

### sph (openmp)
  The simple n^2 SPH simulation (https://github.com/olcf/SPH_Simple)

### split (cuda)
  The split operation in radix sort (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### spm (cuda)
  Image registration calculations for the statistical parametric mapping (SPM) system (http://mri.ee.ntust.edu.tw/cuda/)

### spmm (cuda)
  Library-based sparse matrix sparse matrix product (https://docs.nvidia.com/cuda/cusparse/index.html)

### sptrsv (cuda)
  A thread-Level synchronization-free sparse triangular solver (https://github.com/JiyaSu/CapelliniSpTRSV)

### srad (opencl)
  SRAD (version 1) in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### ss (opencl)
  String search (https://github.com/OpenCL/AMD_APP_samples)

### ssim (cuda)
  Compute structual similarity index measure (https://github.com/VIDILabs/instantvnr)

### sssp (opencl)
  The single-source shortest path (https://github.com/chai-benchmarks/chai)

### stddev (cuda)
  Standard deviation (https://github.com/rapidsai/raft)

### stencil1d (cuda)
  1D stencil (https://www.olcf.ornl.gov/wp-content/uploads/2019/12/02-CUDA-Shared-Memory.pdf)

### stencil3d (cuda)
  3D stencil (https://github.com/LLNL/cardioid)

### streamcluster (opencl)
  Streamcluster in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### stsg (cuda)
  Spatial-temporal Savitzky-Golay method for reconstructing high-quality NDVI time series (https://github.com/HPSCIL/cuSTSG)

### su3 (sycl)
  Lattice QCD SU(3) matrix-matrix multiply microbenchmark (https://gitlab.com/NERSC/nersc-proxies/su3_bench)

### surfel (cuda)
  Surfel rendering (https://github.com/jstraub/cudaPcl)

### svd3x3 (cuda)
  Compute the singular value decomposition of 3x3 matrices (https://github.com/kuiwuchn/3x3_SVD_CUDA)

### sw4ck (cuda)
  SW4 curvilinear kernels are five stencil kernels that account for ~50% of the solution time in SW4 (https://github.com/LLNL/SW4CK)

### swish (cuda)
  The Swish activate functions (https://pytorch.org/)

### tensorAccessor (cuda)
  A demo of tensor accessors in Pytorch (https://pytorch.org/)

### tensorT (cuda)
  Tensor transposition (https://github.com/Jokeren/GPA-Benchmark/tree/master/ExaTENSOR)

### testSNAP (openmp)
  A proxy for the SNAP force calculation in the LAMMPS molecular dynamics package (https://github.com/FitSNAP/TestSNAP)

### thomas (cuda)
  Solve tridiagonal systems of equations using the Thomas algorithm (https://pm.bsc.es/gitlab/run-math/cuThomasBatch/tree/master)

### threadfence (cuda)
  Memory fence function (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

### tissue (cuda)
  Accumulate contributions of tissue source strengths and previous solute levels to current tissue solute levels (https://github.com/secomb/GreensTD19_GPU)

### tonemapping (opencl)
  Tone mapping (https://github.com/OpenCL/AMD_APP_samples)

### tpacf (cuda)
  The 2-point correlation function (https://users.ncsa.illinois.edu/kindr/projects/hpca/index.html)

### tqs (cuda)
  Simulation of a task queue system (https://github.com/chai-benchmarks/chai)

### triad (opencl)
  Triad in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### tridiagonal (opencl)
  Matrix solvers for large number of small independent tridiagonal linear systems(http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### tsa (cuda)
  Trotter-Suzuki approximation (https://bitbucket.org/zzzoom/trottersuzuki/src/master/)

### tsp (cuda)
  Solving the symmetric traveling salesman problem with iterative hill climbing (https://userweb.cs.txstate.edu/~burtscher/research/TSP_GPU/) 

### unfold (cuda)
  Unfold the view of original tensor as slices (https://pytorch.org/)
  
### urng (opencl)
  Uniform random noise generator (https://github.com/OpenCL/AMD_APP_samples)
  
### vanGenuchten (cuda)
  Genuchten conversion of soil moisture and pressure (https://github.com/HydroComplexity/Dhara)

### vmc (cuda)
  Computes expectation values (6D integrals) associated with the helium atom (https://github.com/wadejong/Summer-School-Materials/tree/master/Examples/vmc)

### vol2col (cuda)
  Volume-to-column transform (https://pytorch.org/)

### vote (cuda)
  Demonstrate the usage of the vote intrinsics (https://github.com/NVIDIA/cuda-samples/)

### warpsort (cuda)
  Sort small numbers (https://github.com/facebookarchive/fbcuda)

### wedford (cuda)
  Compute mean and variance using the Welford algorithm (https://github.com/hpcaitech/ColossalAI)

### winograd (opencl)
  Winograd convolution (https://github.com/ChenyangZhang-cs/iMLBench)

### wlcpow (cuda)
  Compute spring forces in a worm-like chain model with a power function (https://github.com/AnselGitAccount/USERMESO-2.0)

### word2vec (cuda)
  Implementation of word2vec with Continuous Bag-of-Words (https://github.com/cudabigdata/word2vec_cuda)

### wordcount (cuda)
  Count the number of words in a text (https://github.com/NVIDIA/thrust/blob/main/examples/)

### wsm5 (cuda)
  Parallel weather and research forecast single moment 5-class (https://github.com/gpgpu-sim/ispass2009-benchmarks/tree/master/wp)

### wyllie (cuda)
  List ranking with Wyllie's algorithm (Rehman, M. & Kothapalli, Kishore & Narayanan, P.. (2009). Fast and Scalable List Ranking on the GPU. Proceedings of the International Conference on Supercomputing. 235-243. 10.1145/1542275.1542311.)

### xlqc (cuda)
  Hartree-Fock self-consistent-field (SCF) calculation of H2O (https://github.com/recoli/XLQC) 

### xsbench (opencl)
  A proxy application for full neutron transport application like OpenMC (https://github.com/ANL-CESAR/XSBench/)

### zerocopy (cuda)
  Kernels may read and write directly to pinned system memory from a user perspective (https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleZeroCopy)

### zeropoint (cuda)
  Find zero-points and scales in quantization (https://pytorch.org/)

### zmddft (cuda)
  3D complex FFT in a 256^3 cube (https://github.com/spiral-software/fftx)

### zoom (cuda)
  Zoom in and zoom out an image (https://github.com/rapidsai/cucim)

## Developer
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
Abhishek Bagusetty, Andrey Alekseenko, Anton Gorshkov, Beau Johnston, Bernhard Esslinger, Bert de Jong, Chengjian Liu, Chris Knight, David Oro, Douglas Franz, Edson Borin, Gabriell Araujo, Georgi Mirazchiyski, Henry Gabb, Hugh Delaney, Ian Karlin, Istvan Reguly, Jack Kirk, Jason Lau, Jeff Hammond, Jenny Chen, Jianxin Qiu, Jakub Chlanda, Jiya Su, John Tramm, Ju Zheng, Junchao Zhang, Kinman Lei, Martin Burtscher, Matthias Noack, Michael Kruse, Michel Migdal, Mike Franusich, Mike Giles, Mikhail Dvorskiy, Mohammed Alser, Muhammad Haseeb, Muaaz Awan, Nevin Liber, Nicholas Miller, Pavel Samolysov, Pedro Valero Lara, Piotr Różański, Rahulkumar Gayatri, Shaoyi Peng, Steffen Larsen, Rafal Bielski, Robert Harrison, Robin Kobus, Rod Burns, Rodrigo Vimieiro, Romanov Vlad, Tadej Ciglarič, Thomas Applencourt, Tiago Carneiro, Timmie Smith, Tobias Baumann, Usman Roshan, Wayne Joubert, Ye Luo, Yongbin Gu, Zhe Chen

Codeplay<sup>®</sup> and Intel<sup>®</sup> for their contributions to the oneAPI ecosystem   

The project uses resources at the Intel<sup>®</sup> DevCloud, the Chameleon testbed supported by the National Science Foundation, the Argonne Leadership Computing Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357, and the Experimental Computing Laboratory (ExCL) at Oak Ridge National Laboratory supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.
