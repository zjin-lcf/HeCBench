#ifndef _SU3_HPP
#define _SU3_HPP
// Adapted from su3.h in MILC version 7

#include <complex>
struct fsu3_matrix { std::complex<float> e[3][3]; } ;
struct fsu3_vector { std::complex<float> c[3]; } ;
struct dsu3_matrix { std::complex<double> e[3][3]; } ;
struct dsu3_vector { std::complex<double> c[3]; } ;


#if (PRECISION==1)
  #define su3_matrix    fsu3_matrix
  #define su3_vector    fsu3_vector
  #define Real          float
  #define Complx        std::complex<float>
#else
  #define su3_matrix    dsu3_matrix
  #define su3_vector    dsu3_vector
  #define Real          double
  #define Complx        std::complex<double>
#endif  // PRECISION

#endif  // _SU3_HPP

