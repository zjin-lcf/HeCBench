/*
 * GPU-accelerated AIDW interpolation algorithm 
 *
 * Implemented with / without CUDA Shared Memory
 *
 * By Dr.Gang Mei
 *
 * Created on 2015.11.06, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * Revised on 2015.12.14, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * 
 * Related publications:
 *  1) "Evaluating the Power of GPU Acceleration for IDW Interpolation Algorithm"
 *     http://www.hindawi.com/journals/tswj/2014/171574/
 *  2) "Accelerating Adaptive IDW Interpolation Algorithm on a Single GPU"
 *     http://arxiv.org/abs/1511.02186
 *
 * License: http://creativecommons.org/licenses/by/4.0/
 */

#include <cstdio>
#include <cstdlib>     
#include <vector>
#include <cmath>
#include "common.h"
#include "reference.h"

// Calculate the power parameter, and then weighted interpolating
// Without using shared memory 
void AIDW_Kernel(
    const float *__restrict dx, 
    const float *__restrict dy,
    const float *__restrict dz,
    const int dnum,
    const float *__restrict ix,
    const float *__restrict iy,
          float *__restrict iz,
    const int inum,
    const float area,
    const float *__restrict avg_dist,
    nd_item<1> &item) 

{
  int tid = item.get_global_id(0);
  if(tid < inum) {
    float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 1.f;

    float r_obs = avg_dist[tid];                // The observed average nearest neighbor distance
    float r_exp = 0.5f / sycl::sqrt(dnum / area); // The expected nearest neighbor distance for a random pattern
    float R_S0 = r_obs / r_exp;                 // The nearest neighbor statistic

    // Normalize the R(S0) measure such that it is bounded by 0 and 1 by a fuzzy membership function 
    float u_R = 0.f;
    if(R_S0 >= R_min) u_R = 0.5f-0.5f * sycl::cos(3.1415926f / R_max * (R_S0 - R_min));
    if(R_S0 >= R_max) u_R = 1.f;

    // Determine the appropriate distance-decay parameter alpha by a triangular membership function
    // Adaptive power parameter: a (alpha)
    if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
    if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if(u_R>0.9f && u_R<=1.f)  alpha = a5;
    alpha *= 0.5f; // Half of the power

    // Weighted average
    for(int j = 0; j < dnum; j++) {
      dist = (ix[tid] - dx[j]) * (ix[tid] - dx[j]) + (iy[tid] - dy[j]) * (iy[tid] - dy[j]) ;
      t = 1.f / sycl::pow(dist, alpha);  sum += t;  z += dz[j] * t;
    }
    iz[tid] = z / sum;
  }
}

// Calculate the power parameter, and then weighted interpolating
// With using shared memory (Tiled version of the stage 2)
void AIDW_Kernel_Tiled(
    const float *__restrict dx, 
    const float *__restrict dy,
    const float *__restrict dz,
    const int dnum,
    const float *__restrict ix,
    const float *__restrict iy,
          float *__restrict iz,
    const int inum,
    const float area,
    const float *__restrict avg_dist,
          float *__restrict sdx,
          float *__restrict sdy,
          float *__restrict sdz,
    nd_item<1> &item) 
{
  int tid = item.get_global_id(0);
  if (tid >= inum) return;

  float dist = 0.f, t = 0.f, alpha = 0.f;

  int part = (dnum - 1) / BLOCK_SIZE;
  int m, e;

  float sum_up = 0.f;
  float sum_dn = 0.f;   
  float six_s, siy_s;

  float r_obs = avg_dist[tid];               //The observed average nearest neighbor distance
  float r_exp = 0.5f / sycl::sqrt(dnum / area); // The expected nearest neighbor distance for a random pattern
  float R_S0 = r_obs / r_exp;                //The nearest neighbor statistic

  float u_R = 0.f;
  if(R_S0 >= R_min) u_R = 0.5f-0.5f * sycl::cos(3.1415926f / R_max * (R_S0 - R_min));
  if(R_S0 >= R_max) u_R = 1.f;

  // Determine the appropriate distance-decay parameter alpha by a triangular membership function
  // Adaptive power parameter: a (alpha)
  if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
  if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
  if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
  if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
  if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
  if(u_R>0.9f && u_R<=1.f)  alpha = a5;
  alpha *= 0.5f; // Half of the power

  float six_t = ix[tid];
  float siy_t = iy[tid];
  int lid = item.get_local_id(0);
  for(m = 0; m <= part; m++) {  // Weighted Sum  
    int num_threads = sycl::min(BLOCK_SIZE, dnum - BLOCK_SIZE *m);
    if (lid < num_threads) {
      sdx[lid] = dx[lid + BLOCK_SIZE * m];
      sdy[lid] = dy[lid + BLOCK_SIZE * m];
      sdz[lid] = dz[lid + BLOCK_SIZE * m];
    }
    item.barrier(access::fence_space::local_space);

    for(e = 0; e < BLOCK_SIZE; e++) {
      six_s = six_t - sdx[e];
      siy_s = siy_t - sdy[e];
      dist = (six_s * six_s + siy_s * siy_s);
      t = 1.f / (sycl::pow(dist, alpha));  sum_dn += t;  sum_up += t * sdz[e];
    }
  }
  iz[tid] = sum_up / sum_dn;
}

int main(int argc, char *argv[])
{
  const int numk = atoi(argv[1]);    // number of points (unit: 1K)
  const int check = atoi(argv[2]);  // do check for small problem sizes

  const int dnum = numk * 1024;
  const int inum = dnum;

  // Area of planar region
  const float width = 2000, height = 2000;
  const float area = width * height;

  std::vector<float> dx(dnum), dy(dnum), dz(dnum);
  std::vector<float> avg_dist(dnum);
  std::vector<float> ix(inum), iy(inum), iz(inum);
  std::vector<float> h_iz(inum);

  srand(123);
  for(int i = 0; i < dnum; i++)
  {
    dx[i] = rand()/(float)RAND_MAX * 1000;
    dy[i] = rand()/(float)RAND_MAX * 1000;
    dz[i] = rand()/(float)RAND_MAX * 1000;
  }

  for(int i = 0; i < inum; i++)
  {
    ix[i] = rand()/(float)RAND_MAX * 1000;
    iy[i] = rand()/(float)RAND_MAX * 1000;
    iz[i] = 0.f;
  }

  for(int i = 0; i < dnum; i++)
  {
    avg_dist[i] = rand()/(float)RAND_MAX * 3;
  }

  printf("Size = : %d K \n", numk);
  printf("dnum = : %d\ninum = : %d\n", dnum, inum);

  if (check) {
    printf("Verification enabled\n");
    reference (dx.data(), dy.data(), dz.data(), dnum, ix.data(), 
               iy.data(), h_iz.data(), inum, area, avg_dist.data());
  } else {
    printf("Verification disabled\n");
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);
  buffer<float, 1> d_dx (dx.data(), dnum);
  buffer<float, 1> d_dy (dy.data(), dnum);
  buffer<float, 1> d_dz (dz.data(), dnum);
  buffer<float, 1> d_avg_dist (avg_dist.data(), dnum);
  buffer<float, 1> d_ix (ix.data(), inum);
  buffer<float, 1> d_iy (iy.data(), inum);
  buffer<float, 1> d_iz (iz.data(), inum);

  range<1> gws ((inum + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  // Weighted Interpolate using AIDW

  for (int i = 0; i < 100; i++) {
    q.submit([&] (handler &cgh) {
      auto dx = d_dx.get_access<sycl_read>(cgh);
      auto dy = d_dy.get_access<sycl_read>(cgh);
      auto dz = d_dz.get_access<sycl_read>(cgh);
      auto ix = d_ix.get_access<sycl_read>(cgh);
      auto iy = d_iy.get_access<sycl_read>(cgh);
      auto iz = d_iz.get_access<sycl_discard_write>(cgh);
      auto avg_dist = d_avg_dist.get_access<sycl_read>(cgh);
      cgh.parallel_for<class aidw>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        AIDW_Kernel(dx.get_pointer(), 
                    dy.get_pointer(),
                    dz.get_pointer(),
                    dnum,
                    ix.get_pointer(),
                    iy.get_pointer(),
                    iz.get_pointer(),
                    inum,
                    area,
                    avg_dist.get_pointer(),
                    item);
      });
    });
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_iz.get_access<sycl_read>(cgh);
    cgh.copy(acc, iz.data());
  }).wait();

  if (check) {
    bool ok = verify (iz.data(), h_iz.data(), inum, EPS);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  for (int i = 0; i < 100; i++) {
    q.submit([&] (handler &cgh) {
      auto dx = d_dx.get_access<sycl_read>(cgh);
      auto dy = d_dy.get_access<sycl_read>(cgh);
      auto dz = d_dz.get_access<sycl_read>(cgh);
      auto ix = d_ix.get_access<sycl_read>(cgh);
      auto iy = d_iy.get_access<sycl_read>(cgh);
      auto iz = d_iz.get_access<sycl_discard_write>(cgh);
      auto avg_dist = d_avg_dist.get_access<sycl_read>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> sdx(BLOCK_SIZE, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> sdy(BLOCK_SIZE, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> sdz(BLOCK_SIZE, cgh);
      cgh.parallel_for<class aidw_tiled>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        AIDW_Kernel_Tiled(dx.get_pointer(), 
                          dy.get_pointer(),
                          dz.get_pointer(),
                          dnum,
                          ix.get_pointer(),
                          iy.get_pointer(),
                          iz.get_pointer(),
                          inum,
                          area,
                          avg_dist.get_pointer(),
                          sdx.get_pointer(),
                          sdy.get_pointer(),
                          sdz.get_pointer(),
                          item);
      });
    });
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_iz.get_access<sycl_read>(cgh);
    cgh.copy(acc, iz.data());
  }).wait();

  if (check) {
    bool ok = verify (iz.data(), h_iz.data(), inum, EPS);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  return 0;
}
