//////////////////////////////////////////////////////////////////
//                                                              //
// This software was written by Mike Giles in 2007 based on     //
// C code written by Zhao and Glasserman at Columbia University //
//                                                              //
// It is copyright University of Oxford, and provided under     //
// the terms of the BSD3 license:                               //
// https://opensource.org/licenses/BSD-3-Clause                 //
//                                                              //
// It is provided along with an informal report on              //
// https://people.maths.ox.ac.uk/~gilesm/cuda_old.html          //
//                                                              //
// Note: this was written for CUDA 1.0 and optimised for        //
// execution on an NVIDIA 8800 GTX GPU                          //
//                                                              //
// Mike Giles, 29 April 2021                                    //
//                                                              //
//////////////////////////////////////////////////////////////////

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// parameters for device execution

#define BLOCK_SIZE 64
#define GRID_SIZE 1500

// parameters for LIBOR calculation

#define NN 80
#define NMAT 40
#define L2_SIZE 3280 //NN*(NMAT+1)
#define NOPT 15
#define NPATH 96000

/* Monte Carlo LIBOR path calculation */

void path_calc(float *L, 
                          const float *z, 
                          const float *lambda, 
                          const float delta,
                          const int Nmat, 
                          const int N)
{
  int   i, n;
  float sqez, lam, con1, v, vrat;

  for(n=0; n<Nmat; n++) {
    sqez = sycl::sqrt((float)delta) * z[n];
    v = 0.f;

    for (i=n+1; i<N; i++) {
      lam  = lambda[i-n-1];
      con1 = delta*lam;
      v += (con1 * L[i]) / (1.f + delta * L[i]);
      vrat = sycl::exp(con1 * v + lam * (sqez - 0.5f * con1));
      L[i] = L[i]*vrat;
    }
  }
}


/* forward path calculation storing data
   for subsequent reverse path calculation */

void path_calc_b1(float *L, 
                             const float *z, 
                             float *L2,
                             const float *lambda,
                             const float delta,
                             const int Nmat,
                             const int N
)
{
  int   i, n;
  float sqez, lam, con1, v, vrat;

  for (i=0; i<N; i++) L2[i] = L[i];
   
  for(n=0; n<Nmat; n++) {
    sqez = sycl::sqrt((float)delta) * z[n];
    v = 0.f;

    for (i=n+1; i<N; i++) {
      lam  = lambda[i-n-1];
      con1 = delta*lam;
      v += (con1 * L[i]) / (1.f + delta * L[i]);
      vrat = sycl::exp(con1 * v + lam * (sqez - 0.5f * con1));
      L[i] = L[i]*vrat;

      // store these values for reverse path //
      L2[i+(n+1)*N] = L[i];
    }
  }
}


/* reverse path calculation of deltas using stored data */

void path_calc_b2(float *L_b, 
                             const float *z, 
                             const float *L2, 
                             const float *lambda, 
                             const float delta,
                             const int Nmat,
                             const int N)
{
  int   i, n;
  float faci, v1;

  for (n=Nmat-1; n>=0; n--) {
    v1 = 0.f;
    for (i=N-1; i>n; i--) {
      v1    += lambda[i-n-1]*L2[i+(n+1)*N]*L_b[i];
      faci = delta / (1.f + delta * L2[i + n * N]);
      L_b[i] = L_b[i] * L2[i + (n + 1) * N] / L2[i + n * N] +
               v1 * lambda[i - n - 1] * faci * faci;
    }
  }
}

/* calculate the portfolio value v, and its sensitivity to L */
/* hand-coded reverse mode sensitivity */

float portfolio_b(float *L, 
                             float *L_b,
                             const float *lambda, 
                             const   int *maturities, 
                             const float *swaprates, 
                             const float delta,
                             const int Nmat,
                             const int N,
                             const int Nopt)
{
  int   m, n;
  float b, s, swapval,v;
  float B[NMAT], S[NMAT], B_b[NMAT], S_b[NMAT];

  b = 1.f;
  s = 0.f;
  for (m=0; m<N-Nmat; m++) {
    n    = m + Nmat;
    b = b / (1.f + delta * L[n]);
    s    = s + delta*b;
    B[m] = b;
    S[m] = s;
  }

  v = 0.f;

  for (m=0; m<N-Nmat; m++) {
    B_b[m] = 0.f;
    S_b[m] = 0.f;
  }

  for (n=0; n<Nopt; n++){
    m = maturities[n] - 1;
    swapval = B[m] + swaprates[n]*S[m] - 1.f;
    if (swapval<0) {
      v     += -100.f*swapval;
      S_b[m] += -100.f*swaprates[n];
      B_b[m] += -100.f;
    }
  }

  for (m=N-Nmat-1; m>=0; m--) {
    n = m + Nmat;
    B_b[m] += delta*S_b[m];
    L_b[n] = -B_b[m] * B[m] * delta / (1.f + delta * L[n]);
    if (m>0) {
      S_b[m-1] += S_b[m];
      B_b[m - 1] += B_b[m] / (1.f + delta * L[n]);
    }
  }

  // apply discount //

  b = 1.f;
  for (n=0; n<Nmat; n++) b = b/(1.f+delta*L[n]);

  v = b*v;

  for (n=0; n<Nmat; n++){
    L_b[n] = -v*delta/(1.f+delta*L[n]);
  }

  for (n=Nmat; n<N; n++){
    L_b[n] = b*L_b[n];
  }

  return v;
}


/* calculate the portfolio value v */

float portfolio(float *L,
                           const float *lambda, 
                           const   int *maturities, 
                           const float *swaprates, 
                           const float delta,
                           const int Nmat,
                           const int N,
                           const int Nopt)
{
  int   n, m, i;
  float v, b, s, swapval, B[40], S[40];
	
  b = 1.f;
  s = 0.f;

  for(n=Nmat; n<N; n++) {
    b = b/(1.f+delta*L[n]);
    s = s + delta*b;
    B[n-Nmat] = b;
    S[n-Nmat] = s;
  }

  v = 0.f;

  for(i=0; i<Nopt; i++){
    m = maturities[i] - 1;
    swapval = B[m] + swaprates[i]*S[m] - 1.f;
    if(swapval<0)
      v += -100.f*swapval;
  }

  // apply discount //

  b = 1.f;
  for (n=0; n<Nmat; n++) b = b/(1.f+delta*L[n]);

  v = b*v;

  return v;
}


void Pathcalc_Portfolio_KernelGPU(float *d_v, 
                                             float *d_Lb,
                                             const float *lambda, 
                                             const   int *maturities, 
                                             const float *swaprates, 
                                             const float delta,
                                             const int Nmat,
                                             const int N,
                                             const int Nopt,
                                             sycl::nd_item<3> item_ct1)
{
  const int tid = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
  const int threadN =
      item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);

  int   i,path;
  float L[NN], L2[L2_SIZE], z[NN];
  float *L_b = L;
  
  /* Monte Carlo LIBOR path calculation*/

  for(path = tid; path < NPATH; path += threadN){
    // initialise the data for current thread
    for (i=0; i<N; i++) {
      // for real application, z should be randomly generated
      z[i] = 0.3f;
      L[i] = 0.05f;
    }
    path_calc_b1(L, z, L2, lambda, delta, Nmat, N);
    d_v[path] = portfolio_b(L, L_b, lambda, maturities, swaprates, delta, Nmat, N, Nopt);
    path_calc_b2(L_b, z, L2, lambda, delta, Nmat, N);
    d_Lb[path] = L_b[NN-1];
  }
}


void Pathcalc_Portfolio_KernelGPU2(float *d_v, 
                                              const float *lambda, 
                                              const   int *maturities, 
                                              const float *swaprates, 
                                              const float delta,
                                              const int Nmat,
                                              const int N,
                                              const int Nopt,
                                              sycl::nd_item<3> item_ct1)
{
  const int tid = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
  const int threadN =
      item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);

  int   i, path;
  float L[NN], z[NN];
  
  /* Monte Carlo LIBOR path calculation*/

  for(path = tid; path < NPATH; path += threadN){
    // initialise the data for current thread
    for (i=0; i<N; i++) {
      // for real application, z should be randomly generated
      z[i] = 0.3f;
      L[i] = 0.05f;
    }	   
    path_calc(L, z, lambda, delta, Nmat, N);
    d_v[path] = portfolio(L, lambda, maturities, swaprates, delta, Nmat, N, Nopt);
  }
}

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // 'h_' prefix - CPU (host) memory space

  float  *h_v, *h_Lb, h_lambda[NN], h_delta=0.25f;
  int     h_N=NN, h_Nmat=NMAT, h_Nopt=NOPT, i;
  int     h_maturities[] = {4,4,4,8,8,8,20,20,20,28,28,28,40,40,40};
  float   h_swaprates[]  = {.045f,.05f,.055f,.045f,.05f,.055f,.045f,.05f,
                            .055f,.045f,.05f,.055f,.045f,.05f,.055f };
  double  v, Lb; 

  // 'd_' prefix - GPU (device) memory space

  float *d_v;
  float *d_Lb;
  float *d_swaprates;
  float *d_lambda;
    int *d_maturities;

  for (i=0; i<NN; i++) h_lambda[i] = 0.2f;

  h_v      = (float *)malloc(sizeof(float)*NPATH);
  h_Lb     = (float *)malloc(sizeof(float)*NPATH);

  d_maturities = (int *)dpct::dpct_malloc(sizeof(h_maturities));

  d_swaprates = (float *)dpct::dpct_malloc(sizeof(h_swaprates));

  d_lambda = (float *)dpct::dpct_malloc(sizeof(h_lambda));

  d_v = (float *)dpct::dpct_malloc(sizeof(float) * NPATH);

  d_Lb = (float *)dpct::dpct_malloc(sizeof(float) * NPATH);

  // Execute GPU kernel -- no Greeks

  sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
  sycl::range<3> dimGrid(1, 1, GRID_SIZE);

  dpct::dpct_memcpy(d_maturities, h_maturities, sizeof(h_maturities),
                    dpct::host_to_device);
  dpct::dpct_memcpy(d_swaprates, h_swaprates, sizeof(h_swaprates),
                    dpct::host_to_device);
  dpct::dpct_memcpy(d_lambda, h_lambda, sizeof(h_lambda), dpct::host_to_device);

  // Launch the device computation threads
  for (int i = 0; i < 100; i++)
    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
  {
    dpct::buffer_t d_v_buf_ct0 = dpct::get_buffer(d_v);
    dpct::buffer_t d_lambda_buf_ct1 = dpct::get_buffer(d_lambda);
    dpct::buffer_t d_maturities_buf_ct2 = dpct::get_buffer(d_maturities);
    dpct::buffer_t d_swaprates_buf_ct3 = dpct::get_buffer(d_swaprates);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_v_acc_ct0 =
          d_v_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto d_lambda_acc_ct1 =
          d_lambda_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto d_maturities_acc_ct2 =
          d_maturities_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
      auto d_swaprates_acc_ct3 =
          d_swaprates_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                         Pathcalc_Portfolio_KernelGPU2(
                             (float *)(&d_v_acc_ct0[0]),
                             (const float *)(&d_lambda_acc_ct1[0]),
                             (const int *)(&d_maturities_acc_ct2[0]),
                             (const float *)(&d_swaprates_acc_ct3[0]), h_delta,
                             h_Nmat, h_N, h_Nopt, item_ct1);
                       });
    });
  }

  // Read back GPU results and compute average
  dpct::dpct_memcpy(h_v, d_v, sizeof(float) * NPATH, dpct::device_to_host);

  v = 0.0;
  for (i=0; i<NPATH; i++) v += h_v[i];
  v = v / NPATH;

  if (fabs(v - 224.323) > 1e-3) printf("Expected: 224.323 Actual %15.3f\n", v);

  // Execute GPU kernel -- Greeks

  // Launch the device computation threads
  for (int i = 0; i < 100; i++)
    /*
    DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
  {
    dpct::buffer_t d_v_buf_ct0 = dpct::get_buffer(d_v);
    dpct::buffer_t d_Lb_buf_ct1 = dpct::get_buffer(d_Lb);
    dpct::buffer_t d_lambda_buf_ct2 = dpct::get_buffer(d_lambda);
    dpct::buffer_t d_maturities_buf_ct3 = dpct::get_buffer(d_maturities);
    dpct::buffer_t d_swaprates_buf_ct4 = dpct::get_buffer(d_swaprates);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_v_acc_ct0 =
          d_v_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Lb_acc_ct1 =
          d_Lb_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto d_lambda_acc_ct2 =
          d_lambda_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
      auto d_maturities_acc_ct3 =
          d_maturities_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
      auto d_swaprates_acc_ct4 =
          d_swaprates_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                         Pathcalc_Portfolio_KernelGPU(
                             (float *)(&d_v_acc_ct0[0]),
                             (float *)(&d_Lb_acc_ct1[0]),
                             (const float *)(&d_lambda_acc_ct2[0]),
                             (const int *)(&d_maturities_acc_ct3[0]),
                             (const float *)(&d_swaprates_acc_ct4[0]), h_delta,
                             h_Nmat, h_N, h_Nopt, item_ct1);
                       });
    });
  }

  // Read back GPU results and compute average
  dpct::dpct_memcpy(h_v, d_v, sizeof(float) * NPATH, dpct::device_to_host);
  dpct::dpct_memcpy(h_Lb, d_Lb, sizeof(float) * NPATH, dpct::device_to_host);

  v = 0.0;
  for (i=0; i<NPATH; i++) v += h_v[i];
  v = v / NPATH;

  Lb = 0.0;
  for (i=0; i<NPATH; i++) Lb += h_Lb[i];
  Lb = Lb / NPATH;

  if (fabs(v - 224.323) > 1e-3) printf("Expected: 224.323 Actual %15.3f\n", v);
  if (fabs(Lb - 21.348) > 1e-3) printf("Expected:  21.348 Actual %15.3f\n", Lb);

  // Release GPU memory

  dpct::dpct_free(d_v);
  dpct::dpct_free(d_Lb);
  dpct::dpct_free(d_maturities);
  dpct::dpct_free(d_swaprates);
  dpct::dpct_free(d_lambda);

  // Release CPU memory

  free(h_v);
  free(h_Lb);

  return 0;
}

