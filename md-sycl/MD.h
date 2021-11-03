#ifndef MD_H__
#define MD_H__

#ifdef SINGLE_PRECISION
#define POSVECTYPE cl::sycl::float4
#define FORCEVECTYPE cl::sycl::float4
#define FPTYPE float
#else
#define POSVECTYPE cl::sycl::double4
#define FORCEVECTYPE cl::sycl::double4
#define FPTYPE double
#endif

// Problem Constants
static const FPTYPE cutsq        = 16.0; // Square of cutoff distance
static const int    maxNeighbors = 128;  // Max number of nearest neighbors
static const FPTYPE domainEdge   = 20.0; // Edge length of the cubic domain
static const FPTYPE lj1          = 1.5;  // LJ constants
static const FPTYPE lj2          = 2.0;

#endif
