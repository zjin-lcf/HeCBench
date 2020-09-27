#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef _SU3_H
#define _SU3_H
/* Adapted from su3.h in MILC version 7 */

/* generic precision complex number definition */
/* specific for float complex */
typedef struct dpct_type_9f9e8b {
  float real;
  float imag; 
} fcomplex;  

/* specific for double complex */
typedef struct dpct_type_e0f020 {
   double real;
   double imag;
} dcomplex;

typedef struct dpct_type_89bfbf { fcomplex e[3][3]; } fsu3_matrix;
typedef struct dpct_type_c0041e { fcomplex c[3]; } fsu3_vector;

typedef struct dpct_type_9f01ed { dcomplex e[3][3]; } dsu3_matrix;
typedef struct dpct_type_ac5cb0 { dcomplex c[3]; } dsu3_vector;

#if (PRECISION==1)
  #define su3_matrix    fsu3_matrix
  #define su3_vector    fsu3_vector
  #define Real          float
  #define Complx        fcomplex
#else
  #define su3_matrix    dsu3_matrix
  #define su3_vector    dsu3_vector
  #define Real          double
  #define Complx        dcomplex
#endif  /* PRECISION */

/*  c += a * b */
#define CMULSUM(a,b,c) { (c).real += (a).real*(b).real - (a).imag*(b).imag; \
                         (c).imag += (a).real*(b).imag + (a).imag*(b).real; }
/*  c = a * b */
#define CMUL(a,b,c) { (c).real = (a).real*(b).real - (a).imag*(b).imag; \
                      (c).imag = (a).real*(b).imag + (a).imag*(b).real; }
/*  a += b    */
#define CSUM(a,b) { (a).real += (b).real; (a).imag += (b).imag; }

#endif  /* _SU3_H */

