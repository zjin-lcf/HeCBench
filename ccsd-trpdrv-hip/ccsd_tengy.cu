#include <stdio.h>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 16

__global__ 
void ccsd_kernel(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                 const double * __restrict__ f2n,    const double * __restrict__ f2t,
                 const double * __restrict__ f3n,    const double * __restrict__ f3t,
                 const double * __restrict__ f4n,    const double * __restrict__ f4t,
                 const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                 const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                 const double * __restrict__ eorb,   const double eaijk,
                 double * __restrict__ emp4i, double * __restrict__ emp5i,
                 double * __restrict__ emp4k, double * __restrict__ emp5k,
                 const int ncor, const int nocc, const int nvir)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;

  if (b < nvir && c < nvir) {

    const double denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

    // nvir < 10000 so this should never overflow
    const int bc = b+c*nvir;
    const int cb = c+b*nvir;

    const double f1nbc = f1n[bc];
    const double f1tbc = f1t[bc];
    const double f1ncb = f1n[cb];
    const double f1tcb = f1t[cb];

    const double f2nbc = f2n[bc];
    const double f2tbc = f2t[bc];
    const double f2ncb = f2n[cb];
    const double f2tcb = f2t[cb];

    const double f3nbc = f3n[bc];
    const double f3tbc = f3t[bc];
    const double f3ncb = f3n[cb];
    const double f3tcb = f3t[cb];

    const double f4nbc = f4n[bc];
    const double f4tbc = f4t[bc];
    const double f4ncb = f4n[cb];
    const double f4tcb = f4t[cb];

    atomicAdd(emp4i , denom * (f1tbc+f1ncb+f2tcb+f3nbc+f4ncb) * (f1tbc-f2tbc*2-f3tbc*2+f4tbc)
                      - denom * (f1nbc+f1tcb+f2ncb+f3ncb) * (f1tbc*2-f2tbc-f3tbc+f4tbc*2)
                      + denom * 3 * (f1nbc*(f1nbc+f3ncb+f4tcb*2) +f2nbc*f2tcb+f3nbc*f4tbc));

    atomicAdd(emp4k , denom * (f1nbc+f1tcb+f2ncb+f3tbc+f4tcb) * (f1nbc-f2nbc*2-f3nbc*2+f4nbc)
                      - denom * (f1tbc+f1ncb+f2tcb+f3tcb) * (f1nbc*2-f2nbc-f3nbc+f4nbc*2)
                      + denom * 3 * (f1tbc*(f1tbc+f3tcb+f4ncb*2) +f2tbc*f2ncb+f3tbc*f4nbc));

    const double t1v1b = t1v1[b];
    const double t1v2b = t1v2[b];

    const double dintx1c = dintx1[c];
    const double dintx2c = dintx2[c];
    const double dintc1c = dintc1[c];
    const double dintc2c = dintc2[c];

    atomicAdd(emp5i, denom * t1v1b * dintx1c * (f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
                     +(f3nbc+f4tbc+f1ncb)*4)
                     + denom * t1v1b * dintc1c * (f1nbc+f4nbc+f1tcb -(f2nbc+f3nbc+f2tcb)*2));
    atomicAdd(emp5k, denom * t1v2b * dintx2c * (f1nbc+f2tbc+f4tcb -(f3nbc+f4tbc+f2tcb +f1tbc+f2nbc+f3tcb)*2
                     +(f3tbc+f4nbc+f1tcb)*4)
                     + denom * t1v2b * dintc2c * (f1tbc+f4tbc+f1ncb -(f2tbc+f3tbc+f2ncb)*2));
  }
}

void ccsd_tengy_gpu(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                    const double * __restrict__ f2n,    const double * __restrict__ f2t,
                    const double * __restrict__ f3n,    const double * __restrict__ f3t,
                    const double * __restrict__ f4n,    const double * __restrict__ f4t,
                    const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                    const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                    const double * __restrict__ eorb,   const double eaijk,
                    double * __restrict__ emp4i_, double * __restrict__ emp5i_,
                    double * __restrict__ emp4k_, double * __restrict__ emp5k_,
                    const int ncor, const int nocc, const int nvir)
{
  double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

  double *d_f1n, *d_f2n, *d_f3n, *d_f4n;
  double *d_f1t, *d_f2t, *d_f3t, *d_f4t;
  double *d_dintc1, *d_dintc2, *d_dintx1, *d_dintx2;
  double *d_t1v1, *d_t1v2, *d_eorb;
  double *d_emp5i, *d_emp4i, *d_emp5k, *d_emp4k;
  hipMalloc((void**)&d_f1n, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_f2n, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_f3n, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_f4n, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_f1t, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_f2t, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_f3t, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_f4t, nvir*nvir*sizeof(double));
  hipMalloc((void**)&d_dintc1, nvir*sizeof(double));
  hipMalloc((void**)&d_dintc2, nvir*sizeof(double));
  hipMalloc((void**)&d_dintx1, nvir*sizeof(double));
  hipMalloc((void**)&d_dintx2, nvir*sizeof(double));
  hipMalloc((void**)&d_t1v1, nvir*sizeof(double));
  hipMalloc((void**)&d_t1v2, nvir*sizeof(double));
  hipMalloc((void**)&d_eorb, (ncor+nocc+nvir)*sizeof(double));
  hipMalloc((void**)&d_emp5i, sizeof(double));
  hipMalloc((void**)&d_emp4i, sizeof(double));
  hipMalloc((void**)&d_emp5k, sizeof(double));
  hipMalloc((void**)&d_emp4k, sizeof(double));

  hipMemcpy(d_f1n, f1n, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f2n, f2n, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f3n, f3n, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f4n, f4n, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f1t, f1t, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f2t, f2t, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f3t, f3t, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f4t, f4t, nvir*nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_dintc1, dintc1, nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_dintc2, dintc2, nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_dintx1, dintx1, nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_dintx2, dintx2, nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_t1v1, t1v1, nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_t1v2, t1v2, nvir*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_eorb, eorb, (ncor+nocc+nvir)*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_emp5i, &emp5i, sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_emp4i, &emp4i, sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_emp5k, &emp5k, sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_emp4k, &emp4k, sizeof(double), hipMemcpyHostToDevice);


  hipLaunchKernelGGL(ccsd_kernel, dim3( (nvir+BLOCK_SIZE-1) / BLOCK_SIZE, (nvir+BLOCK_SIZE-1) / BLOCK_SIZE ), dim3( BLOCK_SIZE, BLOCK_SIZE ), 0, 0, 
        d_f1n,
        d_f1t,
        d_f2n,
        d_f2t,
        d_f3n,
        d_f3t,
        d_f4n,
        d_f4t,
        d_dintc1,
        d_dintx1,
        d_t1v1,
        d_dintc2,
        d_dintx2,
        d_t1v2,
        d_eorb, 
        eaijk,
        d_emp4i, 
        d_emp5i, 
        d_emp4k,
        d_emp5k, 
        ncor, nocc, nvir);

#ifdef DEBUG
  // make the host block until the device is finished
  hipDeviceSynchronize();

  // check for error
  hipError_t error = hipGetLastError();
  if(error != hipSuccess)
  {
    printf("CUDA error: %s\n", hipGetErrorString(error));
  }
#endif

  hipMemcpy(&emp5i, d_emp5i, sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy(&emp4i, d_emp4i, sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy(&emp5k, d_emp5k, sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy(&emp4k, d_emp4k, sizeof(double), hipMemcpyDeviceToHost);

  hipFree(d_f1n);
  hipFree(d_f2n);
  hipFree(d_f3n);
  hipFree(d_f4n);
  hipFree(d_f1t);
  hipFree(d_f2t);
  hipFree(d_f3t);
  hipFree(d_f4t);
  hipFree(d_dintc1);
  hipFree(d_dintc2);
  hipFree(d_dintx1);
  hipFree(d_dintx2);
  hipFree(d_t1v1);
  hipFree(d_t1v2);
  hipFree(d_eorb);
  hipFree(d_emp5i);
  hipFree(d_emp4i);
  hipFree(d_emp5k);
  hipFree(d_emp4k);

  *emp4i_ = emp4i;
  *emp4k_ = emp4k;
  *emp5i_ = emp5i;
  *emp5k_ = emp5k;
}
