/*
  Accumulate contributions of tissue source
  strengths "qt" and previous solute levels "ctprev" to tissue solute levels "ct".
  Each tissue point is assigned one or more threads: step is the number of threads
  This spreads it over more threads.
  TWS September 2014
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"

void reference(
    const   int *d_tisspoints,
    const float *d_gtt,
    const float *d_gbartt,
          float *d_ct,
    const float *d_ctprev,
    const float *d_qt,
    int nnt, int nntDev, int step, int isp)
{
  for (int i = 0; i < step * nnt; i++) {
    int jtp,ixyz,ix,iy,iz,jx,jy,jz,istep;
    int nnt2 = 2*nnt;
    float p = 0.f;

    int itp = i/step;
    int itp1 = i%step;
    if(itp < nnt) {
      ix = d_tisspoints[itp];
      iy = d_tisspoints[itp+nnt];
      iz = d_tisspoints[itp+nnt2];
      for(jtp=itp1; jtp<nnt; jtp+=step){
        jx = d_tisspoints[jtp];
        jy = d_tisspoints[jtp+nnt];
        jz = d_tisspoints[jtp+nnt2];
        ixyz = abs(jx-ix) + abs(jy-iy) + abs(jz-iz) + (isp-1)*nntDev;
        p += d_gtt[ixyz]*d_ctprev[jtp] + d_gbartt[ixyz]*d_qt[jtp];
      }
      if(itp1 == 0) d_ct[itp] = p;
    }

    for(istep=1; istep<step; istep++)
      if(itp1 == istep && itp < nnt) d_ct[itp] += p;
  }
}

void tissue(
    nd_item<1> &item,
    const   int *__restrict d_tisspoints,
    const float *__restrict d_gtt,
    const float *__restrict d_gbartt,
          float *__restrict d_ct,
    const float *__restrict d_ctprev,
    const float *__restrict d_qt,
    int nnt, int nntDev, int step, int isp)
{
  int jtp,ixyz,ix,iy,iz,jx,jy,jz,istep;
  int nnt2 = 2*nnt;
  float p = 0.f;

  const int i = item.get_global_id(0);
  const int itp = i/step;
  const int itp1 = i%step;
  if(itp < nnt) {
    ix = d_tisspoints[itp];
    iy = d_tisspoints[itp+nnt];
    iz = d_tisspoints[itp+nnt2];
    for(jtp = itp1; jtp < nnt; jtp += step) {
      jx = d_tisspoints[jtp];
      jy = d_tisspoints[jtp+nnt];
      jz = d_tisspoints[jtp+nnt2];
      ixyz = sycl::abs(jx-ix) + sycl::abs(jy-iy) + sycl::abs(jz-iz) + (isp-1)*nntDev;
      p += d_gtt[ixyz]*d_ctprev[jtp] + d_gbartt[ixyz]*d_qt[jtp];
    }
    if(itp1 == 0) d_ct[itp] = p;
  }
  // d_ct is incremented in sequence from the needed threads
  for(istep=1; istep<step; istep++)
    if(itp1 == istep && itp < nnt) d_ct[itp] += p;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <dimension of a 3D grid> <repeat>\n", argv[0]);
    return 1;
  }

  const int dim = atoi(argv[1]);
  if (dim > 32) {
    printf("Maximum dimension is 32\n");
    return 1;
  }
  const int repeat = atoi(argv[2]);

  const int nnt = dim * dim * dim;
  const int nntDev = 32*32*32;  // maximum number of tissue points
  const int nsp = 2;

    int* h_tisspoints = (int*) malloc (3*nntDev*sizeof(int));
  float* h_gtt = (float*) malloc (nsp*nntDev*sizeof(float));
  float* h_gbartt = (float*) malloc (nsp*nntDev*sizeof(float));
  float* h_ct = (float*) malloc (nntDev*sizeof(float));
  float* h_ctprev = (float*) malloc (nntDev*sizeof(float));
  float* h_qt = (float*) malloc (nntDev*sizeof(float));
  float* h_ct_gold = (float*) malloc (nntDev*sizeof(float));

  // bound the distance between any two 3D points 
  for (int i = 0; i < 3 * nntDev; i++) {
    h_tisspoints[i] = rand() % (nntDev / 3);
  }
  for (int i = 0; i < nsp * nntDev; i++) {
    h_gtt[i] = rand() / (float)RAND_MAX;
    h_gbartt[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < nntDev; i++) {
    h_ct[i] = h_ct_gold[i] = 0;
    h_ctprev[i] = rand() / (float)RAND_MAX;
    h_qt[i] = rand() / (float)RAND_MAX;
  }

  int step = 4; //a power of two 

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<  int, 1> d_tisspoints (h_tisspoints, 3*nntDev);
  buffer<float, 1> d_gtt (h_gtt, nsp*nntDev);
  buffer<float, 1> d_gbartt (h_gbartt, nsp*nntDev);
  buffer<float, 1> d_ct (h_ct, nntDev);
  buffer<float, 1> d_ctprev (h_ctprev, nntDev);
  buffer<float, 1> d_qt (h_qt, nntDev);
  d_ct.set_final_data(nullptr);

  range<1> lws (256);
  range<1> gws ((step*nnt + 255) / 256 * 256);

  // quick verification and warmup
  for (int i = 0; i < 2; i++) {
    q.submit([&] (handler &cgh) {
      auto tp = d_tisspoints.get_access<sycl_read>(cgh);
      auto gt = d_gtt.get_access<sycl_read>(cgh);
      auto gb = d_gbartt.get_access<sycl_read>(cgh);
      auto ct = d_ct.get_access<sycl_read_write>(cgh);
      auto cp = d_ctprev.get_access<sycl_read>(cgh);
      auto qt = d_qt.get_access<sycl_read>(cgh);
      cgh.parallel_for<class warmup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        tissue(item,
          tp.get_pointer(),gt.get_pointer(),gb.get_pointer(),ct.get_pointer(),
          cp.get_pointer(),qt.get_pointer(),nnt,nntDev,step,1);
      });
    });

    q.submit([&] (handler &cgh) {
      auto tp = d_tisspoints.get_access<sycl_read>(cgh);
      auto gt = d_gtt.get_access<sycl_read>(cgh);
      auto gb = d_gbartt.get_access<sycl_read>(cgh);
      auto ct = d_ct.get_access<sycl_read_write>(cgh);
      auto cp = d_ctprev.get_access<sycl_read>(cgh);
      auto qt = d_qt.get_access<sycl_read>(cgh);
      cgh.parallel_for<class warmup2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        tissue(item,
          tp.get_pointer(),gt.get_pointer(),gb.get_pointer(),ct.get_pointer(),
          cp.get_pointer(),qt.get_pointer(),nnt,nntDev,step,2);
      });
    });
  }

  for (int i = 0; i < 2; i++) {
    reference(h_tisspoints,h_gtt,h_gbartt,h_ct_gold,h_ctprev,h_qt,nnt,nntDev,step,1);
    reference(h_tisspoints,h_gtt,h_gbartt,h_ct_gold,h_ctprev,h_qt,nnt,nntDev,step,2);
  }

  bool ok = true;
  q.submit([&] (handler &cgh) {
    auto acc = d_ct.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_ct);
  }).wait();

  for (int i = 0; i < nntDev; i++) {
    if (fabsf(h_ct[i] - h_ct_gold[i]) > 1e-2) {
      printf("@%d: %f %f\n", i, h_ct[i], h_ct_gold[i]);
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  // timing kernel execution
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto tp = d_tisspoints.get_access<sycl_read>(cgh);
      auto gt = d_gtt.get_access<sycl_read>(cgh);
      auto gb = d_gbartt.get_access<sycl_read>(cgh);
      auto ct = d_ct.get_access<sycl_read_write>(cgh);
      auto cp = d_ctprev.get_access<sycl_read>(cgh);
      auto qt = d_qt.get_access<sycl_read>(cgh);
      cgh.parallel_for<class timing>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        tissue(item,
          tp.get_pointer(),gt.get_pointer(),gb.get_pointer(),ct.get_pointer(),
          cp.get_pointer(),qt.get_pointer(),nnt,nntDev,step,1);
      });
    });

    q.submit([&] (handler &cgh) {
      auto tp = d_tisspoints.get_access<sycl_read>(cgh);
      auto gt = d_gtt.get_access<sycl_read>(cgh);
      auto gb = d_gbartt.get_access<sycl_read>(cgh);
      auto ct = d_ct.get_access<sycl_read_write>(cgh);
      auto cp = d_ctprev.get_access<sycl_read>(cgh);
      auto qt = d_qt.get_access<sycl_read>(cgh);
      cgh.parallel_for<class timing2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        tissue(item,
          tp.get_pointer(),gt.get_pointer(),gb.get_pointer(),ct.get_pointer(),
          cp.get_pointer(),qt.get_pointer(),nnt,nntDev,step,2);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  }

  free(h_tisspoints);
  free(h_gtt);
  free(h_gbartt);
  free(h_ct);
  free(h_ct_gold);
  free(h_ctprev);
  free(h_qt);

  return 0;
}
