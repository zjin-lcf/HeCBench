#include <sycl/sycl.hpp>

double *chebyshev1 ( int n );
double *chebyshev2 ( int n );
double *chebyshev3 ( int n );
double *chebyshev4 ( int n );
double *equidistant1 ( int n );
double *equidistant2 ( int n );
double *equidistant3 ( int n );
double *fejer1 ( int n );
double *fejer2 ( int n );
double lebesgue_constant ( sycl::queue &q, int n, double x[], int nfun, double xfun[] );
double lebesgue_function ( sycl::queue &q, int n, double x[], int nfun, double xfun[] );
double *r8vec_linspace_new ( int n, double a, double b );
double r8vec_max ( int n, double r8vec[] );
void r8vec_print ( int n, double a[], const char *title );
void timestamp ( );
