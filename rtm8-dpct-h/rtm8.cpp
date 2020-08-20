#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define nt 30
#define nx 680
#define ny 134
#define nz 450

#include "mysecond.c"

inline int indexTo1D(int x, int y, int z){
  return x + y*nx + z*nx*ny;
}

void rtm8_cpu(float* vsq, float* current_s, float* current_r, float* next_s, float* next_r, float* image, float* a, size_t N)
{
#ifdef _OPENMP
  #pragma omp parallel for collapse(3)
#endif
  for (int z = 4; z < nz - 4; z++) {
    for (int y = 4; y < ny - 4; y++) {
      for (int x = 4; x < nx - 4; x++) {
        float div =
          a[0] * current_s[indexTo1D(x,y,z)] +
          a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
              current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
              current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
          a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
              current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
              current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
          a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
              current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
              current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
          a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
              current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
              current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

        next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)] - next_s[indexTo1D(x,y,z)]
          + vsq[indexTo1D(x,y,z)]*div;
        div =
          a[0] * current_r[indexTo1D(x,y,z)] +
          a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
              current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
              current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
          a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
              current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
              current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
          a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
              current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
              current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
          a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
              current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
              current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

        next_r[indexTo1D(x,y,z)] = 2 * current_r[indexTo1D(x,y,z)]
          - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)] * div;

        image[indexTo1D(x,y,z)] = next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
      }
    }
  }
}
  

 void
rtm8(float* vsq, float* current_s, float* current_r, float* next_s, float* next_r, float* image, float* a, size_t N,
     sycl::nd_item<3> item_ct1)
{
        unsigned x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
        unsigned y = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                     item_ct1.get_local_id(1);
        unsigned z = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) +
                     item_ct1.get_local_id(0);
  float div;
  if ((4 <= x && x < (nx - 4) ) && (4 <= y && y < (ny - 4)) && (4 <= z && z < (nz - 4))){
    div =
      a[0] * current_s[indexTo1D(x,y,z)] +
      a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
          current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
          current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
      a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
          current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
          current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
      a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
          current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
          current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
      a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
          current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
          current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

    next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)] - next_s[indexTo1D(x,y,z)]
      + vsq[indexTo1D(x,y,z)]*div;
    div =
      a[0] * current_r[indexTo1D(x,y,z)] +
      a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
          current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
          current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
      a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
          current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
          current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
      a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
          current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
          current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
      a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
          current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
          current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

    next_r[indexTo1D(x,y,z)] = 2 * current_r[indexTo1D(x,y,z)]
      - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)] * div;

    image[indexTo1D(x,y,z)] = next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
  }
}


int main() {
  const int ArraySize = nx * ny * nz;
  float* next_s = (float*)malloc(ArraySize * sizeof(float));
  float* current_s = (float*)malloc(ArraySize * sizeof(float));
  float* next_r = (float*)malloc(ArraySize * sizeof(float));
  float* current_r = (float*)malloc(ArraySize * sizeof(float));
  float* vsq = (float*)malloc(ArraySize * sizeof(float));
  float* image_gpu = (float*)malloc(ArraySize * sizeof(float));
  float* image_cpu = (float*)malloc(ArraySize * sizeof(float));

  float a[5];
  double pts, t0, t1, dt, flops, pt_rate, flop_rate, speedup, memory;

  memory = ArraySize*sizeof(float)*6;
  pts = nt;
  pts = pts*(nx-8)*(ny-8)*(nz-8);
  flops = 67*pts;
  printf("memory (MB) = %f\n", memory/1e6);
  printf("pts (billions) = %f\n", pts/1e9);
  printf("Tflops = %f\n", flops/1e12);

  // Initialization of matrix
  a[0] = -1./560.;
  a[1] = 8./315;
  a[2] = -0.2;
  a[3] = 1.6;
  a[4] = -1435./504.;

  for (int z = 0; z < nz; z++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        vsq[indexTo1D(x,y,z)] = 1.0;
        next_s[indexTo1D(x,y,z)] = 0;
        current_s[indexTo1D(x,y,z)] = 1.0;
        next_r[indexTo1D(x,y,z)] = 0;
        current_r[indexTo1D(x,y,z)] = 1.0;
        image_gpu[indexTo1D(x,y,z)] = image_cpu[indexTo1D(x,y,z)] = 0.5;
      }
    }
  }

  t0 = mysecond();
  //allocate and copy matrix to device
  float* vsq_d;
  float* next_s_d;
  float* current_s_d;
  float* next_r_d;
  float* current_r_d;
  float* image_d;
  float* a_d;

        dpct::dpct_malloc(&vsq_d, ArraySize * sizeof(float));
        dpct::dpct_malloc(&next_s_d, ArraySize * sizeof(float));
        dpct::dpct_malloc(&current_s_d, ArraySize * sizeof(float));
        dpct::dpct_malloc(&next_r_d, ArraySize * sizeof(float));
        dpct::dpct_malloc(&current_r_d, ArraySize * sizeof(float));
        dpct::dpct_malloc(&image_d, ArraySize * sizeof(float));
        dpct::dpct_malloc(&a_d, 5 * sizeof(float));

        dpct::dpct_memcpy(vsq_d, vsq, ArraySize * sizeof(float),
                          dpct::host_to_device);
        dpct::dpct_memcpy(next_s_d, next_s, ArraySize * sizeof(float),
                          dpct::host_to_device);
        dpct::dpct_memcpy(current_s_d, current_s, ArraySize * sizeof(float),
                          dpct::host_to_device);
        dpct::dpct_memcpy(next_r_d, next_r, ArraySize * sizeof(float),
                          dpct::host_to_device);
        dpct::dpct_memcpy(current_r_d, current_r, ArraySize * sizeof(float),
                          dpct::host_to_device);
        dpct::dpct_memcpy(image_d, image_gpu, ArraySize * sizeof(float),
                          dpct::host_to_device);
        dpct::dpct_memcpy(a_d, a, 5 * sizeof(float), dpct::host_to_device);

  int groupSize = 16;
  int nx_pad = (nx + groupSize - 1) / groupSize ;
  int ny_pad = (ny + groupSize - 1) / groupSize ;
  int nz_pad = nz;

  // Launch the kernel nt times
  for (int t = 0; t < nt; t++) {
                dpct::buffer_t vsq_d_buf_ct0 = dpct::get_buffer(vsq_d);
                dpct::buffer_t current_s_d_buf_ct1 =
                    dpct::get_buffer(current_s_d);
                dpct::buffer_t next_s_d_buf_ct2 = dpct::get_buffer(next_s_d);
                dpct::buffer_t current_r_d_buf_ct3 =
                    dpct::get_buffer(current_r_d);
                dpct::buffer_t next_r_d_buf_ct4 = dpct::get_buffer(next_r_d);
                dpct::buffer_t image_d_buf_ct5 = dpct::get_buffer(image_d);
                dpct::buffer_t a_d_buf_ct6 = dpct::get_buffer(a_d);
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        auto vsq_d_acc_ct0 =
                            vsq_d_buf_ct0
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);
                        auto current_s_d_acc_ct1 =
                            current_s_d_buf_ct1
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);
                        auto next_s_d_acc_ct2 =
                            next_s_d_buf_ct2
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);
                        auto current_r_d_acc_ct3 =
                            current_r_d_buf_ct3
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);
                        auto next_r_d_acc_ct4 =
                            next_r_d_buf_ct4
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);
                        auto image_d_acc_ct5 =
                            image_d_buf_ct5
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);
                        auto a_d_acc_ct6 =
                            a_d_buf_ct6
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(nz_pad, ny_pad, nx_pad) *
                                    sycl::range<3>(1, groupSize, groupSize),
                                sycl::range<3>(1, groupSize, groupSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                    rtm8((float *)(&vsq_d_acc_ct0[0]),
                                         (float *)(&current_s_d_acc_ct1[0]),
                                         (float *)(&next_s_d_acc_ct2[0]),
                                         (float *)(&current_r_d_acc_ct3[0]),
                                         (float *)(&next_r_d_acc_ct4[0]),
                                         (float *)(&image_d_acc_ct5[0]),
                                         (float *)(&a_d_acc_ct6[0]), ArraySize,
                                         item_ct1);
                            });
                });
  }

  //copy back image value
        dpct::dpct_memcpy(image_gpu, image_d, ArraySize * sizeof(float),
                          dpct::device_to_host);
  t1 = mysecond();
  dt = t1 - t0;

  t0 = mysecond();
  for (int t = 0; t < nt; t++) {
    rtm8_cpu(vsq, current_s, next_s, current_r, next_r, image_cpu, a, ArraySize);
  }
  t1 = mysecond();

  // verification
  for (int i = 0; i < ArraySize; i++)
                if (fabsf(image_cpu[i] - image_gpu[i]) > 0.1) {
      printf("@index %d cpu: %f gpu %f\n", i, image_cpu[i], image_gpu[i]);
      break;
    }

  pt_rate = pts/dt;
  flop_rate = flops/dt;
  printf("dt = %f\n", dt);
  printf("pt_rate (millions/sec) = %f\n", pt_rate/1e6);
  printf("flop_rate (Gflops) = %f\n", flop_rate/1e9);
  printf("speedup over cpu = %f\n", (t1 - t0) / dt);

  //release arrays
  free(vsq);
  free(next_s);
  free(current_s);
  free(next_r);
  free(current_r);
  free(image_cpu);
  free(image_gpu);
        dpct::dpct_free(vsq_d);
        dpct::dpct_free(next_s_d);
        dpct::dpct_free(current_s_d);
        dpct::dpct_free(next_r_d);
        dpct::dpct_free(current_r_d);
        dpct::dpct_free(image_d);
        dpct::dpct_free(a_d);

  return 0;
}

