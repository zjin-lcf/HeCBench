#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"

#define CHUNK_S 4096

typedef struct {
  float x, y, z;
} kdata;

void cmpfhd(const float*__restrict rmu, 
            const float*__restrict imu,
                  float*__restrict rfhd,
                  float*__restrict ifhd,
            const float*__restrict x, 
            const float*__restrict y,
            const float*__restrict z,
            const kdata*__restrict k,
            const int samples,
            const int voxels,
            nd_item<1> &item ) 
{
  int n = item.get_global_id(0);

  if (n < samples) {
    float xn = x[n], yn = y[n], zn = z[n];
    float rfhdn = rfhd[n], ifhdn = ifhd[n];
    for (int m = 0; m < voxels; m++) {
      float e = 2.f * (float)M_PI * 
                (k[m].x * xn + k[m].y * yn + k[m].z * zn);
      float c = sycl::cos(e);
      float s = sycl::sin(e);
      rfhdn += rmu[m] * c - imu[m] * s;
      ifhdn += imu[m] * c + rmu[m] * s;
    }
    rfhd[n] = rfhdn, ifhd[n] = ifhdn;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s #samples #voxels\n", argv[0]);
    exit(1);
  }
  const int samples = atoi(argv[1]); // in the order of 100000
  const int voxels = atoi(argv[2]);  // cube(128)/2097152
  const int sampleSize = samples * sizeof(float);
  const int voxelSize = voxels * sizeof(float);

  float *h_rmu = (float*) malloc (voxelSize);
  float *h_imu = (float*) malloc (voxelSize);
  float *h_kx = (float*) malloc (voxelSize);
  float *h_ky = (float*) malloc (voxelSize);
  float *h_kz = (float*) malloc (voxelSize);
  kdata *h_k = (kdata*) malloc (voxels * sizeof(kdata));

  float *h_rfhd = (float*) malloc (sampleSize);
  float *h_ifhd = (float*) malloc (sampleSize);
  float *h_x = (float*) malloc (sampleSize);
  float *h_y = (float*) malloc (sampleSize);
  float *h_z = (float*) malloc (sampleSize);

  // For device results
  float *rfhd = (float*) malloc (sampleSize);
  float *ifhd = (float*) malloc (sampleSize);

  srand(2);
  for (int i = 0; i < samples; i++) {
    h_rfhd[i] = (float)i/samples;
    h_ifhd[i] = (float)i/samples;
    h_x[i] = 0.3f + (rand()%2 ? 0.1 : -0.1);
    h_y[i] = 0.2f + (rand()%2 ? 0.1 : -0.1);
    h_z[i] = 0.1f + (rand()%2 ? 0.1 : -0.1);
  }

  for (int i = 0; i < voxels; i++) {
    h_rmu[i] = (float)i/voxels;
    h_imu[i] = (float)i/voxels;
    h_k[i].x = h_kx[i] = 0.1f + (rand()%2 ? 0.1 : -0.1);
    h_k[i].y = h_ky[i] = 0.2f + (rand()%2 ? 0.1 : -0.1);
    h_k[i].z = h_kz[i] = 0.3f + (rand()%2 ? 0.1 : -0.1);
  }

  printf("Run FHd on a device\n");

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_rmu (h_rmu, voxels);
  buffer<float, 1> d_imu (h_imu, voxels);
  buffer<float, 1> d_rfhd (h_rfhd, samples);
  buffer<float, 1> d_ifhd (h_ifhd, samples);
  buffer<float, 1> d_x (h_x, samples);
  buffer<float, 1> d_y (h_y, samples);
  buffer<float, 1> d_z (h_z, samples);
  buffer<kdata, 1> d_k (CHUNK_S);
  d_rfhd.set_final_data(nullptr);
  d_ifhd.set_final_data(nullptr);

  const int ntpb = 256;
  const int nblks = (samples + ntpb - 1) / ntpb * ntpb;
  range<1> gws (nblks);
  range<1> lws (ntpb);

  int c = CHUNK_S;
  int nchunks = (voxels + c - 1) / c;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nchunks; i++) {
    if (i == nchunks - 1) {
      c = voxels - CHUNK_S * i;
    }

    q.submit([&] (handler &cgh) {
      auto acc = d_k.get_access<sycl_discard_write>(cgh, range<1>(c));
      cgh.copy(&h_k[i * CHUNK_S], acc);
    }).wait();

    q.submit([&] (handler &cgh) {
      auto rmu = d_rmu.get_access<sycl_read>(cgh, range<1>(c), id<1>(i*CHUNK_S));
      auto imu = d_imu.get_access<sycl_read>(cgh, range<1>(c), id<1>(i*CHUNK_S));
      auto rfhd = d_rfhd.get_access<sycl_read_write>(cgh);
      auto ifhd = d_ifhd.get_access<sycl_read_write>(cgh);
      auto x = d_x.get_access<sycl_read>(cgh);
      auto y = d_y.get_access<sycl_read>(cgh);
      auto z = d_z.get_access<sycl_read>(cgh);
      auto k = d_k.get_access<sycl_read, sycl_cmem>(cgh, range<1>(c));
      cgh.parallel_for<class fhd>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {

        cmpfhd( rmu.get_pointer(),
                imu.get_pointer(),
                rfhd.get_pointer(), 
                ifhd.get_pointer(), 
                x.get_pointer(),
                y.get_pointer(),
                z.get_pointer(),
                k.get_pointer(),
                samples, c, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device execution time %f (s)\n", time * 1e-9f);

  q.submit([&] (handler &cgh) {
    auto acc = d_rfhd.get_access<sycl_read>(cgh);
    cgh.copy(acc, rfhd);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_ifhd.get_access<sycl_read>(cgh);
    cgh.copy(acc, ifhd);
  });

  q.wait();

  printf("Computing root mean square error between host and device results.\n");
  printf("This will take a while..\n");

  #pragma omp parallel for 
  for (int n = 0; n < samples; n++) {
    float r = h_rfhd[n];
    float i = h_ifhd[n];
    #pragma omp parallel for simd reduction(+:r,i)
    for (int m = 0; m < voxels; m++) {
      float e = 2.f * (float)M_PI * 
                (h_kx[m] * h_x[n] + h_ky[m] * h_y[n] + h_kz[m] * h_z[n]);
      float c = cosf(e);
      float s = sinf(e);
      r += h_rmu[m] * c - h_imu[m] * s;
      i += h_imu[m] * c + h_rmu[m] * s;
    }
    h_rfhd[n] = r;
    h_ifhd[n] = i;   
  }

  float err = 0.f;
  for (int i = 0; i < samples; i++) {
    err += (h_rfhd[i] - rfhd[i]) * (h_rfhd[i] - rfhd[i]) +
           (h_ifhd[i] - ifhd[i]) * (h_ifhd[i] - ifhd[i]) ;
  }
  printf("RMSE = %f\n", sqrtf(err / (2*samples)));
 
  free(h_rmu);
  free(h_imu);
  free(h_kx);
  free(h_ky);
  free(h_kz);
  free(h_k);
  free(h_rfhd);
  free(h_ifhd);
  free(rfhd);
  free(ifhd);
  free(h_x);
  free(h_y);
  free(h_z);

  return 0;
   
}
