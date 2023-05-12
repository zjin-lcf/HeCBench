#ifndef INT_H
#define INT_H

#include <math.h>
#include <stdlib.h>
#if defined SHORT

typedef short int_t;
#define conversion_factor 10.0f
#define RF "%d"
#define RCONST(x) x
#define ALMOST_ZERO RCONST(1e-7)
#define MATHFN(x) x##f

#elif defined INT

typedef int int_t;
#define conversion_factor 100.0f
#define RF "%d"
#define RCONST(x) x
#define ALMOST_ZERO RCONST(1e-15)
#define MATHFN(x) x

#elif defined LONG

typedef long int int_t;
#define conversion_factor 100.0f
#define RF "%d"
#define RCONST(x) x##L
#define ALMOST_ZERO RCONST(1e-19)
#define MATHFN(x) x##l

#else

#error Must define one of SHORT, INT, or LONG

#endif

#ifdef __CUDACC__
#include <sycl/sycl.hpp>
#define LOG sycl::log
#define ABS sycl::abs
#else
#define LOG MATHFN(log)
// integer math function
#define ABS abs
#endif



#define STR_TO_INT(x,y) floor(strtof(x,y)*conversion_factor + 0.5f)
#define HALF RCONST(0.5)
#define PI RCONST(3.1415926535897932384626433832795029)
#define INF (14000) //the GPU doesn't like INT_MAX or SHORT_MAX
#define NOT_A_NUMBER 14000 //an arbitrary large number to represent infinity

#endif /* INT_H */
