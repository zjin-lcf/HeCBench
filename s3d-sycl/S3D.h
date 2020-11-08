#ifndef GPU_GLOBAL_H
#define GPU_GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define BLOCK_SIZE   64
#define BLOCK_SIZE2  (2*BLOCK_SIZE)

//alleviate aliasing issues
#define RESTRICT __restrict__

//replace divisions by multiplication with the reciprocal
#define REPLACE_DIV_WITH_RCP 1

#if REPLACE_DIV_WITH_RCP
template <class T1, class T2>
inline T1 DIV(T1 x, T2 y)
{
   return x * (1.0f / y);
}
#else
template <class T1, class T2>
inline T1 DIV(T1 x, T2 y)
{
   return x / y;
}
#endif

// Choose correct intrinsics based on precision
// POW
template<class T>
inline T POW (T in, T in2);

template<>
inline double POW<double>(double in, double in2)
{
    return cl::sycl::pow(in, in2);
}

template<>
inline float POW<float>(float in, float in2)
{
    return cl::sycl::pow(in, in2);
}
// EXP
template<class T>
inline T EXP(T in);

template<>
inline double EXP<double>(double in)
{
    return cl::sycl::exp(in);
}

template<>
inline float EXP<float>(float in)
{
    return cl::sycl::exp(in);
}

// EXP10
template<class T>
inline T EXP10(T in);

template<>
inline double EXP10<double>(double in)
{
    return cl::sycl::exp10(in);
}

template<>
inline float EXP10<float>(float in)
{
    return cl::sycl::exp10(in);
}

// EXP2
template<class T>
inline T EXP2(T in);

template<>
inline double EXP2<double>(double in)
{
    return cl::sycl::exp2(in);
}

template<>
inline float EXP2<float>(float in)
{
    return cl::sycl::exp2(in);
}

// FMAX
template<class T>
inline T MAX(T in, T in2);

template<>
inline double MAX<double>(double in, double in2)
{
    return cl::sycl::fmax(in, in2);
}

template<>
inline float MAX<float>(float in, float in2)
{
    return cl::sycl::fmax(in, in2);
}

// FMIN
template<class T>
inline T MIN(T in, T in2);

template<>
inline double MIN<double>(double in, double in2)
{
    return cl::sycl::fmin(in, in2);
}

template<>
inline float MIN<float>(float in, float in2)
{
    return cl::sycl::fmin(in, in2);
}

// LOG
template<class T>
inline T LOG(T in);

template<>
inline double LOG<double>(double in)
{
    return cl::sycl::log(in);
}

template<>
inline float LOG<float>(float in)
{
    return cl::sycl::log(in);
}

// LOG10
template<class T>
inline T LOG10(T in);

template<>
inline double LOG10<double>(double in)
{
    return cl::sycl::log10(in);
}

template<>
inline float LOG10<float>(float in)
{
    return cl::sycl::log10(in);
}

template <class t1, class t2, class t3, class t4, class t5>
t1 polyx(t1 x, t2 c0, t3 c1, t4 c2, t5 c3)
{
    return (((c3 * x + c2) * x + c1) * x + c0) * x;
}

//Kernel indexing macros
#define N_GP item.get_global_range(0) // number of grid points
#define idx2(p,z) (p[(((z)-1)*(N_GP)) + item.get_global_id(0)])
#define idx(x, y) ((x)[(y)-1])

#define C(q)     idx2(C, q)
#define Y(q)     idx2(Y, q)
#define RF(q)    idx2(RF, q)
#define EG(q)    idx2(EG, q)
#define RB(q)    idx2(RB, q)
#define RKLOW(q) idx2(RKLOW, q)
#define ROP(q)   idx(ROP, q)
#define WDOT(q)  idx2(WDOT, q)
#define RKF(q)   idx2(RKF, q)
#define RKR(q)   idx2(RKR, q)
#define A_DIM    (11)
#define A(b, c)  idx2(A, (((b)*A_DIM)+c) )

// Size macros
// This is the number of floats/doubles per thread for each var

#define C_SIZE               (22)
#define RF_SIZE             (206)
#define RB_SIZE             (206)
#define WDOT_SIZE            (22)
#define RKLOW_SIZE           (21)
#define Y_SIZE               (22)
#define A_SIZE    (A_DIM * A_DIM)
#define EG_SIZE              (32)

#define ROP2(a)  (RKF(a) - RKR (a))

#endif
