#ifndef MD_H__
#define MD_H__

#ifdef SINGLE_PRECISION

#define FPTYPE float
#define POSVECTYPE sycl::float4
#define FORCEVECTYPE sycl::float4

#else

#define FPTYPE double
#define POSVECTYPE sycl::double4
#define FORCEVECTYPE sycl::double4

#endif

// Problem Constants
static const FPTYPE cutsq     = 13.5; // Square of cutoff distance
static const int maxNeighbors = 128;  // Max number of nearest neighbors
static const int domainEdge   = 20;   // Edge length of the cubic domain
static const FPTYPE lj1       = 1.5;  // LJ constants
static const FPTYPE lj2       = 2.0;

#endif
