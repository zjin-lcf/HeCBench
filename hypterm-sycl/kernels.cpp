#include <stdio.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define max(x,y)  ((x) > (y)? (x) : (y))
#define min(x,y)  ((x) < (y)? (x) : (y))
#define ceil(a,b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1)

void check_error (double *ptr, const char* message) {
  if (ptr == nullptr) {
    printf ("Error : %s\n", message);
  }
}

__attribute__ ((always_inline))
void hypterm_1 (sycl::nd_item<3> &item,
                double * __restrict flux_0,
                double * __restrict flux_1,
                double * __restrict flux_2,
                double * __restrict flux_3,
                double * __restrict flux_4,
                const double * __restrict cons_1,
                const double * __restrict cons_2,
                const double * __restrict cons_3,
                const double * __restrict cons_4, 
                const double * __restrict q_1,
                const double * __restrict q_2,
                const double * __restrict q_3,
                const double * __restrict q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(item.get_local_range(2));
  int i0 = (int)(item.get_group(2))*(blockdim_i);
  int i = max (i0, 0) + (int)(item.get_local_id(2));
  int blockdim_j= (int)(item.get_local_range(1));
  int j0 = (int)(item.get_group(1))*(blockdim_j);
  int j = max (j0, 0) + (int)(item.get_local_id(1));
  int blockdim_k= (int)(item.get_local_range(0));
  int k0 = (int)(item.get_group(0))*(blockdim_k);
  int k = max (k0, 0) + (int)(item.get_local_id(0));

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1])-0.2f*(cons_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2])+0.038f*(cons_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3])-0.0035f*(cons_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_1[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]))-0.2f*(cons_1[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]))+0.038f*(cons_1[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]))-0.0035f*(cons_1[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4])))*dxinv0);
  	 flux_2[k*M*N+j*N+i] = -((0.8f*(cons_2[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_2[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_2[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_2[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_2[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_2[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_2[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_2[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_3[k*M*N+j*N+i] = -((0.8f*(cons_3[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_3[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_3[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_3[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_3[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_3[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_3[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_3[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_4[k*M*N+j*N+i] = -((0.8f*(cons_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]))-0.2f*(cons_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]))+0.038f*(cons_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]))-0.0035f*(cons_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4])))*dxinv0);
  } 
}

__attribute__ ((always_inline))
void hypterm_2 (sycl::nd_item<3> &item,
                double * __restrict flux_0,
                double * __restrict flux_1,
                double * __restrict flux_2,
                double * __restrict flux_3,
                double * __restrict flux_4,
                const double * __restrict cons_1,
                const double * __restrict cons_2,
                const double * __restrict cons_3,
                const double * __restrict cons_4, 
                const double * __restrict q_1,
                const double * __restrict q_2,
                const double * __restrict q_3,
                const double * __restrict q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(item.get_local_range(2));
  int i0 = (int)(item.get_group(2))*(blockdim_i);
  int i = max (i0, 0) + (int)(item.get_local_id(2));
  int blockdim_j= (int)(item.get_local_range(1));
  int j0 = (int)(item.get_group(1))*(blockdim_j);
  int j = max (j0, 0) + (int)(item.get_local_id(1));
  int blockdim_k= (int)(item.get_local_range(0));
  int k0 = (int)(item.get_group(0))*(blockdim_k);
  int k = max (k0, 0) + (int)(item.get_local_id(0));

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i])-0.2f*(cons_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i])+0.038f*(cons_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_1[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_1[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_1[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_1[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_1[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_1[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_1[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]))-0.2f*(cons_2[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]))+0.038f*(cons_2[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]))-0.0035f*(cons_2[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i])))*dxinv1;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_3[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_3[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_3[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_3[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_3[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_3[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_3[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]))-0.2f*(cons_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]))+0.038f*(cons_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]))-0.0035f*(cons_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i])))*dxinv2;
  } 
}

__attribute__ ((always_inline))
void hypterm_3 (sycl::nd_item<3> &item,
                double * __restrict flux_0,
                double * __restrict flux_1,
                double * __restrict flux_2,
                double * __restrict flux_3,
                double * __restrict flux_4,
                const double * __restrict cons_1,
                const double * __restrict cons_2,
                const double * __restrict cons_3,
                const double * __restrict cons_4, 
                const double * __restrict q_1,
                const double * __restrict q_2,
                const double * __restrict q_3,
                const double * __restrict q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(item.get_local_range(2));
  int i0 = (int)(item.get_group(2))*(blockdim_i);
  int i = max (i0, 0) + (int)(item.get_local_id(2));
  int blockdim_j= (int)(item.get_local_range(1));
  int j0 = (int)(item.get_group(1))*(blockdim_j);
  int j = max (j0, 0) + (int)(item.get_local_id(1));
  int blockdim_k= (int)(item.get_local_range(0));
  int k0 = (int)(item.get_group(0))*(blockdim_k);
  int k = max (k0, 0) + (int)(item.get_local_id(0));

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i])-0.2f*(cons_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i])+0.038f*(cons_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_1[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_1[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_1[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_1[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_1[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_1[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_1[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_2[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_2[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_2[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_2[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_2[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_2[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_2[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]))-0.2f*(cons_3[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]))+0.038f*(cons_3[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]))-0.0035f*(cons_3[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i])))*dxinv2;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]))-0.2f*(cons_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]))+0.038f*(cons_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]))-0.0035f*(cons_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i])))*dxinv1;
  } 
}

extern "C" void offload (double *h_flux_0, double *h_flux_1, double *h_flux_2, double *h_flux_3, double *h_flux_4, double *h_cons_1, double *h_cons_2, double *h_cons_3, double *h_cons_4, double *h_q_1, double *h_q_2, double *h_q_3, double *h_q_4, double dxinv0, double dxinv1, double dxinv2, int L, int M, int N, int repeat) {

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t vol = L*M*N;
  size_t vol_size = sizeof(double) * vol;

  double *flux_0 = sycl::malloc_device<double>(vol, q);
  check_error (flux_0, "Failed to allocate device memory for flux_0\n");

  double *flux_1 = sycl::malloc_device<double>(vol, q);
  check_error (flux_1, "Failed to allocate device memory for flux_1\n");

  double *flux_2 = sycl::malloc_device<double>(vol, q);
  check_error (flux_2, "Failed to allocate device memory for flux_2\n");

  double *flux_3 = sycl::malloc_device<double>(vol, q);
  check_error (flux_3, "Failed to allocate device memory for flux_3\n");

  double *flux_4 = sycl::malloc_device<double>(vol, q);
  check_error (flux_4, "Failed to allocate device memory for flux_4\n");

  double *cons_1 = sycl::malloc_device<double>(vol, q);
  check_error (cons_1, "Failed to allocate device memory for cons_1\n");
  q.memcpy (cons_1, h_cons_1, vol_size);

  double *cons_2 = sycl::malloc_device<double>(vol, q);
  check_error (cons_2, "Failed to allocate device memory for cons_2\n");
  q.memcpy (cons_2, h_cons_2, vol_size);

  double *cons_3 = sycl::malloc_device<double>(vol, q);
  check_error (cons_3, "Failed to allocate device memory for cons_3\n");
  q.memcpy (cons_3, h_cons_3, vol_size);

  double *cons_4 = sycl::malloc_device<double>(vol, q);
  check_error (cons_4, "Failed to allocate device memory for cons_4\n");
  q.memcpy (cons_4, h_cons_4, vol_size);

  double *q_1 = sycl::malloc_device<double>(vol, q);
  check_error (q_1, "Failed to allocate device memory for q_1\n");
  q.memcpy (q_1, h_q_1, vol_size);

  double *q_2 = sycl::malloc_device<double>(vol, q);
  check_error (q_2, "Failed to allocate device memory for q_2\n");
  q.memcpy (q_2, h_q_2, vol_size);

  double *q_3 = sycl::malloc_device<double>(vol, q);
  check_error (q_3, "Failed to allocate device memory for q_3\n");
  q.memcpy (q_3, h_q_3, vol_size);

  double *q_4 = sycl::malloc_device<double>(vol, q);
  check_error (q_4, "Failed to allocate device memory for q_3\n");
  q.memcpy (q_4, h_q_4, vol_size);

  sycl::range<3> lws (4, 4, 16);
  sycl::range<3> gws (ceil(L, 4)*4, ceil(M, 4)*4, ceil(N, 16)*16);

  long t1 = 0, t2 = 0, t3 = 0;

  for (int i = 0; i < repeat; i++) {
    q.memcpy (flux_0, h_flux_0, vol_size);
    q.memcpy (flux_1, h_flux_1, vol_size);
    q.memcpy (flux_2, h_flux_2, vol_size);
    q.memcpy (flux_3, h_flux_3, vol_size);
    q.memcpy (flux_4, h_flux_4, vol_size);
    q.wait();

    auto start = std::chrono::steady_clock::now();
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k1>(sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        hypterm_1(item, flux_0, flux_1, flux_2, flux_3, flux_4,
                  cons_1, cons_2, cons_3, cons_4,
                  q_1, q_2, q_3, q_4,
                  dxinv0, dxinv1, dxinv2, L, M, N);
      });
    }).wait();
    auto end = std::chrono::steady_clock::now();
    t1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k2>(sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        hypterm_2(item, flux_0, flux_1, flux_2, flux_3, flux_4,
                  cons_1, cons_2, cons_3, cons_4,
                  q_1, q_2, q_3, q_4,
                  dxinv0, dxinv1, dxinv2, L, M, N);
      });
    }).wait();
    end = std::chrono::steady_clock::now();
    t2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k3>(sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        hypterm_3(item, flux_0, flux_1, flux_2, flux_3, flux_4,
                  cons_1, cons_2, cons_3, cons_4,
                  q_1, q_2, q_3, q_4,
                  dxinv0, dxinv1, dxinv2, L, M, N);
      });
    }).wait();
    end = std::chrono::steady_clock::now();
    t3 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time (k1): %f (ms)\n", t1 * 1e-6 / repeat);
  printf("Average kernel execution time (k2): %f (ms)\n", t2 * 1e-6 / repeat);
  printf("Average kernel execution time (k3): %f (ms)\n", t3 * 1e-6 / repeat);

  q.memcpy (h_flux_0, flux_0, vol_size);
  q.memcpy (h_flux_1, flux_1, vol_size);
  q.memcpy (h_flux_2, flux_2, vol_size);
  q.memcpy (h_flux_3, flux_3, vol_size);
  q.memcpy (h_flux_4, flux_4, vol_size);
  q.wait();

  sycl::free(cons_1, q);
  sycl::free(cons_2, q);
  sycl::free(cons_3, q);
  sycl::free(cons_4, q);
  sycl::free(q_1, q);
  sycl::free(q_2, q);
  sycl::free(q_3, q);
  sycl::free(q_4, q);
  sycl::free(flux_0, q);
  sycl::free(flux_1, q);
  sycl::free(flux_2, q);
  sycl::free(flux_3, q);
  sycl::free(flux_4, q);
}
