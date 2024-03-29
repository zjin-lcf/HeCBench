#include <math.h>
#include <hip/hip_runtime.h>

/* double-precision atomic max is defined in HIP
__device__ __forceinline__
double atomicMax(double *address, double val)
{
  unsigned long long ret = __double_as_longlong(*address);
  while(val > __longlong_as_double(ret))
  {
    unsigned long long old = ret;
    if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}
*/

__global__
void kernel (double *__restrict__ lmax,
             double *__restrict__ linterp,
             const double *__restrict__ xfun, 
             const double *__restrict__ x,
             const int n, const int nfun) 
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nfun) return;
  for (int i = 0; i < n; i++ )
    linterp[i*nfun+j] = 1.0;

  for (int i1 = 0; i1 < n; i1++ )
    for (int i2 = 0; i2 < n; i2++ )
      if ( i1 != i2 )
        linterp[i1*nfun+j] = linterp[i1*nfun+j] * ( xfun[j] - x[i2] ) / ( x[i1] - x[i2] );

  double t = 0.0;
  for (int i = 0; i < n; i++ )
    t += fabs ( linterp[i*nfun+j] );

  atomicMax(lmax, t);
}

double lebesgue_function ( int n, double x[], int nfun, double xfun[] )
{
  double lmax = 0.0;

  double *d_max, *d_interp, *d_xfun, *d_x;
  hipMalloc((void**)&d_max, sizeof ( double ) );
  hipMemcpy(d_max, &lmax, sizeof ( double ), hipMemcpyHostToDevice );

  hipMalloc((void**)&d_interp, n * nfun * sizeof ( double ) );

  hipMalloc((void**)&d_x, n * sizeof ( double ) );
  hipMemcpy(d_x, x, n * sizeof ( double ), hipMemcpyHostToDevice );

  hipMalloc((void**)&d_xfun, nfun * sizeof ( double ) );
  hipMemcpy(d_xfun, xfun, nfun * sizeof ( double ), hipMemcpyHostToDevice );

  dim3 grids ((nfun + 255)/256);
  dim3 blocks (256);

  hipLaunchKernelGGL(kernel, grids, blocks, 0, 0, d_max, d_interp, d_xfun, d_x, n, nfun);
  hipMemcpy(&lmax, d_max, sizeof ( double ), hipMemcpyDeviceToHost );

  hipFree(d_max);
  hipFree(d_interp);
  hipFree(d_xfun);
  hipFree(d_x);
  return lmax;
}
