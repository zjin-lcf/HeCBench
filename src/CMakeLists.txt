# src/CMakeLists.txt
# Benchmark discovery and registration

# Add common include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# List of benchmarks to build (will be expanded over time)
# Started with proof-of-concept, now expanding coverage
set(HECBENCH_POC_BENCHMARKS
    # Original POC
    jacobi
    bfs
    softmax
    attention
    # Batch 1: A-F alphabetically (15 benchmarks)
    accuracy
    ace
    adam
    adamw
    aes
    affine
    aidw
    amgmk
    aobench
    backprop
    bilateral
    cfd
    clenergy
    dct8x8
    fft
    # Batch 2: G-L alphabetically (15 benchmarks)
    gaussian
    geodesic
    glu
    gmm
    heartwall
    hmm
    hotspot
    hybridsort
    idivide
    interleave
    inversek2j
    ising
    knn
    lebesgue
    lud
    # Batch 3: M-P alphabetically (11 benchmarks)
    md
    md5hash
    miniDGS
    minimod
    minkowski
    mixbench
    nn
    nw
    particlefilter
    pathfinder
    perplexity
    # Batch 4: Q-Z alphabetically (18 benchmarks - some not found)
    qtclustering
    radixsort
    randomAccess
    remap
    sad
    scan
    scatter
    srad
    sssp
    stencil
    streamcluster
    su3
    sw
    tensor
    transpose
    triad
    tsa
    xsbench
    # Batch 5: A-B benchmarks (20 benchmarks)
    adjacent
    adv
    aligned-types
    all-pairs-distance
    ans
    aop
    asmooth
    assert
    asta
    atan2
    atomicCost
    atomicIntrinsics
    atomicPerf
    atomicReduction
    babelstream
    background-subtract
    bezier-surface
    binomial
    bitonic-sort
    black-scholes
    # Batch 6: B-D benchmarks (16 benchmarks)
    blas-gemm
    burger
    bwt
    channelSum
    chi2
    clink
    cobahh
    compute-score
    convolution1D
    convolutionSeparable
    crc64
    crs
    damage
    degrid
    diamond
    dslash
)

# Include benchmark subdirectories
foreach(BENCHMARK ${HECBENCH_POC_BENCHMARKS})
    # Check which models exist for this benchmark
    foreach(MODEL cuda hip sycl omp)
        set(BENCHMARK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${BENCHMARK}-${MODEL}")
        if(EXISTS "${BENCHMARK_DIR}/CMakeLists.txt")
            add_subdirectory(${BENCHMARK_DIR})
        endif()
    endforeach()
endforeach()

# Create category targets after all benchmarks are registered
create_category_targets()

# Print summary
print_benchmark_summary()
