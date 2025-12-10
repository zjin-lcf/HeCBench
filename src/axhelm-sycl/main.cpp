#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>

#define POLYNOMIAL_DEGREE  7
#define p_Nggeo 7
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_GWJID 0

#include "meshBasis.hpp"

// cpu reference 
#include "axhelmReference.cpp"


dfloat *drandAlloc(int Nelem){

  dfloat *v = (dfloat*) calloc(Nelem, sizeof(dfloat));

  for(int n=0;n<Nelem;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv){

  if (argc<4) {
    printf("Usage: ./axhelm Ndim numElements [nRepetitions]\n");
    return 1;
  }

  const int Ndim = atoi(argv[1]);
  const int Nelements = atoi(argv[2]);
  int Ntests = 1;
  if(argc>=4)
    Ntests = atoi(argv[3]);

  const int Nq = POLYNOMIAL_DEGREE + 1;
  const int Np = Nq*Nq*Nq;
  const int offset = Nelements*Np;


  // build element nodes and operators
  dfloat *rV, *wV, *DrV;
  meshJacobiGQ(0,0,POLYNOMIAL_DEGREE, &rV, &wV);
  meshDmatrix1D(POLYNOMIAL_DEGREE, Nq, rV, &DrV);

  std::cout << "word size: " << sizeof(dfloat) << " bytes\n";

  // populate device arrays
  dfloat *ggeo = drandAlloc(Np*Nelements*p_Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  const dfloat lambda1 = 1.1;
  dfloat *lambda = (dfloat*) calloc(2*offset, sizeof(dfloat));
  for(int i=0; i<offset; i++) {
    lambda[i]        = 1.0;
    lambda[i+offset] = lambda1;
  }

  for(int n=0;n<Ndim;++n){
    dfloat *x = q + n*offset;
    dfloat *Ax = Aq + n*offset; 
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }

  auto start = std::chrono::high_resolution_clock::now();

#ifdef USE_GPU
  sycl::queue Q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue Q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  dfloat *o_ggeo, *o_q, *o_Aq, *o_DrV, *o_lambda;
  o_ggeo = sycl::malloc_device<dfloat>(Np*Nelements*p_Nggeo, Q);
  o_q    = sycl::malloc_device<dfloat>(Ndim*Np*Nelements, Q);
  o_Aq   = sycl::malloc_device<dfloat>(Ndim*Np*Nelements, Q);
  o_DrV  = sycl::malloc_device<dfloat>(Nq*Nq, Q);
  o_lambda = sycl::malloc_device<dfloat>(2*offset, Q);

  Q.memcpy(o_ggeo, ggeo, Np*Nelements*p_Nggeo*sizeof(dfloat));
  Q.memcpy(o_q, q, (Ndim*Np)*Nelements*sizeof(dfloat));
  Q.memcpy(o_DrV, DrV, Nq*Nq*sizeof(dfloat));
  Q.memcpy(o_lambda, lambda, 2*offset*sizeof(dfloat));

  sycl::range<2> global_work_size(8, Nelements*8);
  sycl::range<2> local_work_size(8, 8);

    for(int test = 0; test < Ntests; ++test) {
      if (Ndim > 1)
        Q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<dfloat, 1> s_D(64, cgh);
            sycl::local_accessor<dfloat, 1> s_U(64, cgh);
            sycl::local_accessor<dfloat, 1> s_V(64, cgh);
            sycl::local_accessor<dfloat, 1> s_W(64, cgh);
            sycl::local_accessor<dfloat, 1> s_GUr(64, cgh);
            sycl::local_accessor<dfloat, 1> s_GUs(64, cgh);
            sycl::local_accessor<dfloat, 1> s_GVr(64, cgh);
            sycl::local_accessor<dfloat, 1> s_GVs(64, cgh);
            sycl::local_accessor<dfloat, 1> s_GWr(64, cgh);
            sycl::local_accessor<dfloat, 1> s_GWs(64, cgh);

            cgh.parallel_for<class axhelm_ndim3>(
                sycl::nd_range<2>(global_work_size, local_work_size), [=] (sycl::nd_item<2> item) {
                dfloat r_Ut, r_Vt, r_Wt;
                dfloat r_U[8], r_V[8], r_W[8];
                dfloat r_AU[8], r_AV[8], r_AW[8];
                dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
                dfloat r_lam0, r_lam1;

                int e = item.get_group(1); 
                int j = item.get_local_id(0);
                int i = item.get_local_id(1);

                s_D[j*8+i] = o_DrV[j*8+i];
                const int base = i + j * 8 + e * 512;
                for (int k = 0; k < 8; k++) {
                r_U[k] = o_q[base + k * 8 * 8 + 0 * offset];
                r_V[k] = o_q[base + k * 8 * 8 + 1 * offset];
                r_W[k] = o_q[base + k * 8 * 8 + 2 * offset];
                r_AU[k] = 0;
                r_AV[k] = 0;
                r_AW[k] = 0;
                }
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                const int gbase = e * p_Nggeo * 512 + k * 8 * 8 + j * 8 + i;
                r_G00 = o_ggeo[gbase + p_G00ID * 512];
                r_G01 = o_ggeo[gbase + p_G01ID * 512];
                r_G02 = o_ggeo[gbase + p_G02ID * 512];
                r_G11 = o_ggeo[gbase + p_G11ID * 512];
                r_G12 = o_ggeo[gbase + p_G12ID * 512];
                r_G22 = o_ggeo[gbase + p_G22ID * 512];
                r_GwJ = o_ggeo[gbase + p_GWJID * 512];
                r_lam0 = o_lambda[id + 0 * offset];
                r_lam1 = o_lambda[id + 1 * offset];
                item.barrier(sycl::access::fence_space::local_space);
                s_U[j*8+i] = r_U[k];
                s_V[j*8+i] = r_V[k];
                s_W[j*8+i] = r_W[k];
                r_Ut = 0;
                r_Vt = 0;
                r_Wt = 0;
#pragma unroll 8
                for (int m = 0; m < 8; m++) {
                  dfloat Dkm = s_D[k*8+m];
                  r_Ut += Dkm * r_U[m];
                  r_Vt += Dkm * r_V[m];
                  r_Wt += Dkm * r_W[m];
                }
                item.barrier(sycl::access::fence_space::local_space);
                dfloat Ur = 0, Us = 0;
                dfloat Vr = 0, Vs = 0;
                dfloat Wr = 0, Ws = 0;
#pragma unroll 8
                for (int m = 0; m < 8; m++) {
                  dfloat Dim = s_D[i*8+m];
                  dfloat Djm = s_D[j*8+m];
                  Ur += Dim * s_U[j*8+m];
                  Us += Djm * s_U[m*8+i];
                  Vr += Dim * s_V[j*8+m];
                  Vs += Djm * s_V[m*8+i];
                  Wr += Dim * s_W[j*8+m];
                  Ws += Djm * s_W[m*8+i];
                }
                s_GUr[j*8+i] = r_lam0 * (r_G00 * Ur + r_G01 * Us + r_G02 * r_Ut);
                s_GVr[j*8+i] = r_lam0 * (r_G00 * Vr + r_G01 * Vs + r_G02 * r_Vt);
                s_GWr[j*8+i] = r_lam0 * (r_G00 * Wr + r_G01 * Ws + r_G02 * r_Wt);
                s_GUs[j*8+i] = r_lam0 * (r_G01 * Ur + r_G11 * Us + r_G12 * r_Ut);
                s_GVs[j*8+i] = r_lam0 * (r_G01 * Vr + r_G11 * Vs + r_G12 * r_Vt);
                s_GWs[j*8+i] = r_lam0 * (r_G01 * Wr + r_G11 * Ws + r_G12 * r_Wt);
                r_Ut = r_lam0 * (r_G02 * Ur + r_G12 * Us + r_G22 * r_Ut);
                r_Vt = r_lam0 * (r_G02 * Vr + r_G12 * Vs + r_G22 * r_Vt);
                r_Wt = r_lam0 * (r_G02 * Wr + r_G12 * Ws + r_G22 * r_Wt);
                r_AU[k] += r_GwJ * r_lam1 * r_U[k];
                r_AV[k] += r_GwJ * r_lam1 * r_V[k];
                r_AW[k] += r_GwJ * r_lam1 * r_W[k];
                item.barrier(sycl::access::fence_space::local_space);
                dfloat AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 8
                for (int m = 0; m < 8; m++) {
                  dfloat Dmi = s_D[m*8+i];
                  dfloat Dmj = s_D[m*8+j];
                  dfloat Dkm = s_D[k*8+m];
                  AUtmp += Dmi * s_GUr[j*8+m];
                  AUtmp += Dmj * s_GUs[m*8+i];
                  AVtmp += Dmi * s_GVr[j*8+m];
                  AVtmp += Dmj * s_GVs[m*8+i];
                  AWtmp += Dmi * s_GWr[j*8+m];
                  AWtmp += Dmj * s_GWs[m*8+i];
                  r_AU[m] += Dkm * r_Ut;
                  r_AV[m] += Dkm * r_Vt;
                  r_AW[m] += Dkm * r_Wt;
                }
                r_AU[k] += AUtmp;
                r_AV[k] += AVtmp;
                r_AW[k] += AWtmp;
                }
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                  o_Aq[id + 0 * offset] = r_AU[k];
                  o_Aq[id + 1 * offset] = r_AV[k];
                  o_Aq[id + 2 * offset] = r_AW[k];
                }
            });
        });
      else
        Q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<dfloat, 1> s_D(64, cgh);
            sycl::local_accessor<dfloat, 1> s_q(64, cgh);
            sycl::local_accessor<dfloat, 1> s_Gqr(64, cgh);
            sycl::local_accessor<dfloat, 1> s_Gqs(64, cgh);

            cgh.parallel_for<class axhelm_ndim1>(
                sycl::nd_range<2>(global_work_size, local_work_size), [=] (sycl::nd_item<2> item) {
                dfloat r_qt, r_Gqt, r_Auk;
                dfloat r_q[8];
                dfloat r_Aq[8];
                dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
                dfloat r_lam0, r_lam1;

                int e = item.get_group(1); 
                int j = item.get_local_id(0);
                int i = item.get_local_id(1);

                s_D[j*8+i] = o_DrV[j*8+i];
                const int base = i + j * 8 + e * 512;
                for (int k = 0; k < 8; ++k) {
                r_q[k] = o_q[base + k * 8 * 8];
                r_Aq[k] = 0;
                }
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                const int gbase = e * p_Nggeo * 512 + k * 8 * 8 + j * 8 + i;
                r_G00 = o_ggeo[gbase + p_G00ID * 512];
                r_G01 = o_ggeo[gbase + p_G01ID * 512];
                r_G02 = o_ggeo[gbase + p_G02ID * 512];
                r_G11 = o_ggeo[gbase + p_G11ID * 512];
                r_G12 = o_ggeo[gbase + p_G12ID * 512];
                r_G22 = o_ggeo[gbase + p_G22ID * 512];
                r_GwJ = o_ggeo[gbase + p_GWJID * 512];
                r_lam0 = o_lambda[id + 0 * offset];
                r_lam1 = o_lambda[id + 1 * offset];
                item.barrier(sycl::access::fence_space::local_space);
                s_q[j*8+i] = r_q[k];
                r_qt = 0;
#pragma unroll 8
                for (int m = 0; m < 8; ++m) {
                  r_qt += s_D[k*8+m] * r_q[m];
                }
                item.barrier(sycl::access::fence_space::local_space);
                dfloat qr = 0;
                dfloat qs = 0;
#pragma unroll 8
                for (int m = 0; m < 8; ++m) {
                  qr += s_D[i*8+m] * s_q[j*8+m];
                  qs += s_D[j*8+m] * s_q[m*8+i];
                }
                s_Gqs[j*8+i] = r_lam0 * (r_G01 * qr + r_G11 * qs + r_G12 * r_qt);
                s_Gqr[j*8+i] = r_lam0 * (r_G00 * qr + r_G01 * qs + r_G02 * r_qt);
                r_Gqt = r_lam0 * (r_G02 * qr + r_G12 * qs + r_G22 * r_qt);
                r_Auk = r_GwJ * r_lam1 * r_q[k];
                item.barrier(sycl::access::fence_space::local_space);
#pragma unroll 8
                for (int m = 0; m < 8; ++m) {
                  r_Auk += s_D[m*8+j] * s_Gqs[m*8+i];
                  r_Aq[m] += s_D[k*8+m] * r_Gqt;
                  r_Auk += s_D[m*8+i] * s_Gqr[j*8+m];
                }
                r_Aq[k] += r_Auk;
                item.barrier(sycl::access::fence_space::local_space);
                }
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                  o_Aq[id] = r_Aq[k];
                }
            });
        });
    }
  // store the device results in the 'q' array
  Q.memcpy(q, o_Aq, Ndim*Np*Nelements*sizeof(dfloat)).wait();

  sycl::free(o_ggeo, Q);
  sycl::free(o_q, Q);
  sycl::free(o_Aq, Q);
  sycl::free(o_DrV, Q);
  sycl::free(o_lambda, Q);

  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;


  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = std::fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  std::cout << "Correctness check: maxError = " << maxDiff << "\n";
  free(ggeo);
  free(q);
  free(Aq);
  free(lambda);
  free(rV);
  free(wV);
  free(DrV);

  // print statistics
  const dfloat GDOFPerSecond = Ndim*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*Nelements/elapsed;
  const long long bytesMoved = (Ndim*2*Np+7*Np+2*Np)*sizeof(dfloat); // x, Mx, opa, lambda
  const double bw = bytesMoved*Nelements/elapsed;
  double flopCount = Ndim*Np*12*Nq;
  if(Ndim == 1) flopCount += 22*Np;
  if(Ndim == 3) flopCount += 69*Np;
  double gflops = flopCount*Nelements/elapsed;
  std::cout << " NRepetitions=" << Ntests
    << " Ndim=" << Ndim
    << " N=" << POLYNOMIAL_DEGREE
    << " Nelements=" << Nelements
    << " elapsed time=" << elapsed
    << " GDOF/s=" << GDOFPerSecond
    << " GB/s=" << bw
    << " GFLOPS/s=" << gflops
    << "\n";
  return 0;
}
