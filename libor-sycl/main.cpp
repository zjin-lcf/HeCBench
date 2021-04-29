////////////////////////////////////////////////////////////////////
////                                                              //
//// This software was written by Mike Giles in 2007 based on     //
//// C code written by Zhao and Glasserman at Columbia University //
////                                                              //
//// It is copyright University of Oxford, and provided under     //
//// the terms of the BSD3 license:                               //
//// https://opensource.org/licenses/BSD-3-Clause                 //
////                                                              //
//// It is provided along with an informal report on              //
//// https://people.maths.ox.ac.uk/~gilesm/cuda_old.html          //
////                                                              //
//// Note: this was written for CUDA 1.0 and optimised for        //
//// execution on an NVIDIA 8800 GTX GPU                          //
////                                                              //
//// Mike Giles, 29 April 2021                                    //
////                                                              //
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"

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
    sqez = cl::sycl::sqrt(delta)*z[n];
    v = 0.f;

    for (i=n+1; i<N; i++) {
      lam  = lambda[i-n-1];
      con1 = delta*lam;
      v   += cl::sycl::native::divide(con1*L[i],1.f+delta*L[i]);
      vrat = cl::sycl::native::exp(con1*v + lam*(sqez-0.5f*con1));
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
                  const int N)
{
  int   i, n;
  float sqez, lam, con1, v, vrat;

  for (i=0; i<N; i++) L2[i] = L[i];
   
  for(n=0; n<Nmat; n++) {
    sqez = cl::sycl::sqrt(delta)*z[n];
    v = 0.f;

    for (i=n+1; i<N; i++) {
      lam  = lambda[i-n-1];
      con1 = delta*lam;
      v   += cl::sycl::native::divide(con1*L[i],1.f+delta*L[i]);
      vrat = cl::sycl::native::exp(con1*v + lam*(sqez-0.5f*con1));
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
      faci   = cl::sycl::native::divide(delta,1.f+delta*L2[i+n*N]);
      L_b[i] = L_b[i]*cl::sycl::native::divide(L2[i+(n+1)*N],L2[i+n*N])
              + v1*lambda[i-n-1]*faci*faci;
 
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
    b    = cl::sycl::native::divide(b,1.f+delta*L[n]);
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
    L_b[n]  = -B_b[m]*B[m]*cl::sycl::native::divide(delta,1.f+delta*L[n]);
    if (m>0) {
      S_b[m-1] += S_b[m];
      B_b[m-1] += cl::sycl::native::divide(B_b[m],1.f+delta*L[n]);
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


int main(int argc, char **argv)
{
  // 'h_' prefix - CPU (host) memory space

  float  *h_v, *h_Lb, h_lambda[NN], h_delta=0.25f;
  int     h_N=NN, h_Nmat=NMAT, h_Nopt=NOPT, i;
  int     h_maturities[] = {4,4,4,8,8,8,20,20,20,28,28,28,40,40,40};
  float   h_swaprates[]  = {.045f,.05f,.055f,.045f,.05f,.055f,.045f,.05f,
                            .055f,.045f,.05f,.055f,.045f,.05f,.055f };
  double  v, Lb; 

  // 'd_' prefix - GPU (device) memory space

  for (i=0; i<NN; i++) h_lambda[i] = 0.2f;

  h_v  = (float *)malloc(sizeof(float)*NPATH);
  h_Lb = (float *)malloc(sizeof(float)*NPATH);

  // Execute GPU kernel -- no Greeks


#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);

  buffer<  int,1> d_maturities(h_maturities, NOPT);
  buffer<float,1> d_swaprates(h_swaprates, NOPT);
  buffer<float,1> d_lambda(h_lambda, NN);
  buffer<float,1> d_v(NPATH);
  buffer<float,1> d_Lb(NPATH);

  range<1> gws (GRID_SIZE * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  // Launch the device computation threads
  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto maturities = d_maturities.get_access<sycl_read>(cgh);
      auto swaprates  = d_swaprates.get_access<sycl_read>(cgh);
      auto lambda = d_lambda.get_access<sycl_read>(cgh);
      auto v = d_v.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class porfolio_nogreek>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        const int     tid = item.get_global_id(0);
        const int threadN = item.get_global_range(0);
        int   i, path;
        float L[NN], z[NN];
        
        /* Monte Carlo LIBOR path calculation*/
        for(path = tid; path < NPATH; path += threadN){
          // initialise the data for current thread
          for (i = 0; i < h_N; i++) {
            // for real application, z should be randomly generated
            z[i] = 0.3f;
            L[i] = 0.05f;
          }
          path_calc(L, z, lambda.get_pointer(), h_delta, h_Nmat, h_N);
          v[path] = portfolio(L, lambda.get_pointer(), maturities.get_pointer(), 
                              swaprates.get_pointer(), h_delta, h_Nmat, h_N, h_Nopt);
        }
      });
    });

  // Read back GPU results and compute average
  q.submit([&] (handler &cgh) {
    auto v = d_v.get_access<sycl_read>(cgh);
    cgh.copy(v, h_v);
  }).wait();
  
  v = 0.0;
  for (i=0; i<NPATH; i++) v += h_v[i];
  v = v / NPATH;

  if (std::fabs(v - 224.323) > 1e-3) printf("Expected: 224.323 Actual %15.3f\n", v);

  // Execute GPU kernel -- Greeks

  // Launch the device computation threads
  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto maturities = d_maturities.get_access<sycl_read>(cgh);
      auto swaprates  = d_swaprates.get_access<sycl_read>(cgh);
      auto lambda = d_lambda.get_access<sycl_read>(cgh);
      auto v = d_v.get_access<sycl_discard_write>(cgh);
      auto Lb = d_Lb.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class porfolio_greek>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        const int     tid = item.get_global_id(0);
        const int threadN = item.get_global_range(0);

        int   i,path;
        float L[NN], L2[L2_SIZE], z[NN];
        float *L_b = L;
        
        /* Monte Carlo LIBOR path calculation*/

        for(path = tid; path < NPATH; path += threadN){
          // initialise the data for current thread
          for (i = 0; i < h_N; i++) {
            // for real application, z should be randomly generated
            z[i] = 0.3f;
            L[i] = 0.05f;
          }
          path_calc_b1(L, z, L2, lambda.get_pointer(), h_delta, h_Nmat, h_N);
          v[path] = portfolio_b(L, L_b, lambda.get_pointer(), maturities.get_pointer(), 
                                swaprates.get_pointer(), h_delta, h_Nmat, h_N, h_Nopt);
          path_calc_b2(L_b, z, L2, lambda.get_pointer(), h_delta, h_Nmat, h_N);
          Lb[path] = L_b[NN-1];
        }
      });
    });

  // Read back GPU results and compute average

  q.submit([&] (handler &cgh) {
    auto Lb = d_Lb.get_access<sycl_read>(cgh);
    cgh.copy(Lb, h_Lb);
  });

  q.submit([&] (handler &cgh) {
    auto v = d_v.get_access<sycl_read>(cgh);
    cgh.copy(v, h_v);
  });

  q.wait();

  v = 0.0;
  for (i=0; i<NPATH; i++) v += h_v[i];
  v = v / NPATH;

  Lb = 0.0;
  for (i=0; i<NPATH; i++) Lb += h_Lb[i];
  Lb = Lb / NPATH;

  if (std::fabs(v - 224.323) > 1e-3) printf("Expected: 224.323 Actual %15.3f\n", v);
  if (std::fabs(Lb - 21.348) > 1e-3) printf("Expected:  21.348 Actual %15.3f\n", Lb);

  // Release CPU memory

  free(h_v);
  free(h_Lb);

  return 0;
}

