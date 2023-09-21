#ifndef MD_H__
#define MD_H__

#ifdef SINGLE_PRECISION

#define FPTYPE float
typedef struct __attribute__((__aligned__(16)))
{
  float x; float y; float z; float w; 
} float4;
#define POSVECTYPE float4
#define FORCEVECTYPE float4
#define zero {0.f,0.f,0.f,0.f}

#else

#define FPTYPE double
typedef struct __attribute__((__aligned__(32)))
{
  double x; double y; double z; double w; 
} double4;
#define POSVECTYPE double4
#define FORCEVECTYPE double4
#define zero {0.0,0.0,0.0,0.0}

#endif

// Problem Constants
static const FPTYPE cutsq     = 13.5; // Square of cutoff distance
static const int maxNeighbors = 128;  // Max number of nearest neighbors
static const int domainEdge   = 20;   // Edge length of the cubic domain
static const FPTYPE lj1       = 1.5;  // LJ constants
static const FPTYPE lj2       = 2.0;

#endif
