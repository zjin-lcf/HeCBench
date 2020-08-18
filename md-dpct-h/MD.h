#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef MD_H__
#define MD_H__

#ifdef SINGLE_PRECISION
#define POSVECTYPE sycl::float4
#define FORCEVECTYPE sycl::float4
#define FPTYPE float
#else
#define POSVECTYPE double4
#define FORCEVECTYPE double4
#define FPTYPE double
#endif

// Problem Constants
static const float  cutsq        = 16.0f; // Square of cutoff distance
static const int    maxNeighbors = 128;  // Max number of nearest neighbors
static const double domainEdge   = 20.0; // Edge length of the cubic domain
static const float  lj1          = 1.5;  // LJ constants
static const float  lj2          = 2.0;
static const float  EPSILON      = 0.1f; // Relative Error between CPU/GPU

#endif
