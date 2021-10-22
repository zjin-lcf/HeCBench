#include <math.h>
#include <stdlib.h>
#include <cuda.h>

__global__
void kernel (double *__restrict__ lfun,
             double *__restrict__ linterp,
             const double *__restrict__ xfun, 
             const double *__restrict__ x,
             const int n, const int nfun) 
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nfun) return;
  if ( n == 1 ) {
    lfun[j] = 1.0;
  } else {
    for (int i = 0; i < n; i++ )
      linterp[i*nfun+j] = 1.0;

    for (int i1 = 0; i1 < n; i1++ )
      for (int i2 = 0; i2 < n; i2++ )
        if ( i1 != i2 )
          linterp[i1*nfun+j] = linterp[i1*nfun+j] * ( xfun[j] - x[i2] ) / ( x[i1] - x[i2] );

    double t = 0.0;
    for (int i = 0; i < n; i++ )
      t += fabs ( linterp[i*nfun+j] );

    lfun[j] = t;
  }
}

double *lebesgue_function ( int n, double x[], int nfun, double xfun[] )
{
  double *lfun = ( double * ) malloc ( nfun * sizeof ( double ) );

  double *d_fun, *d_interp, *d_xfun, *d_x;
  cudaMalloc((void**)&d_fun, nfun * sizeof ( double ) );
  cudaMalloc((void**)&d_interp, n * nfun * sizeof ( double ) );
  cudaMalloc((void**)&d_x, n * sizeof ( double ) );
  cudaMalloc((void**)&d_xfun, nfun * sizeof ( double ) );

  cudaMemcpy(d_x, x, n * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy(d_xfun, xfun, nfun * sizeof ( double ), cudaMemcpyHostToDevice );
  
  dim3 grids ((nfun + 255)/256);
  dim3 blocks (256);

  kernel<<<grids, blocks>>> (d_fun, d_interp, d_xfun, d_x, n, nfun);
  cudaMemcpy(lfun, d_fun, nfun * sizeof ( double ), cudaMemcpyDeviceToHost );
  cudaFree(d_fun);
  cudaFree(d_interp);
  cudaFree(d_xfun);
  cudaFree(d_x);
  return lfun;
}
