#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define CHUNK_S 4096

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif


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
            sycl::nd_item<1> &item )
{
  int n = item.get_global_id(0);

  if (n < samples) {
    float xn = ldg(&x[n]), yn = ldg(&y[n]), zn = ldg(&z[n]);
    float rfhdn = ldg(&rfhd[n]), ifhdn = ldg(&ifhd[n]);
    for (int m = 0; m < voxels; m++) {
      float e = 2.f * (float)M_PI * (k[m].x * xn + k[m].y * yn + k[m].z * zn);
      float c = sycl::native::cos(e);
      float s = sycl::native::sin(e);
      rfhdn += ldg(&rmu[m]) * c - ldg(&imu[m]) * s;
      ifhdn += ldg(&imu[m]) * c + ldg(&rmu[m]) * s;
    }
    rfhd[n] = rfhdn, ifhd[n] = ifhdn;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <#samples> <#voxels> <verify>\n", argv[0]);
    exit(1);
  }
  const int samples = atoi(argv[1]); // in the order of 100000
  const int voxels = atoi(argv[2]);  // cube(128)/2097152
  const int verify = atoi(argv[3]);
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
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_rmu = sycl::malloc_device<float>(voxels, q);
  q.memcpy(d_rmu, h_rmu, voxelSize);

  float *d_imu = sycl::malloc_device<float>(voxels, q);
  q.memcpy(d_imu, h_imu, voxelSize);

  float *d_rfhd = sycl::malloc_device<float>(samples, q);
  q.memcpy(d_rfhd, h_rfhd, sampleSize);

  float *d_ifhd = sycl::malloc_device<float>(samples, q);
  q.memcpy(d_ifhd, h_ifhd, sampleSize);

  float *d_x = sycl::malloc_device<float>(samples, q);
  q.memcpy(d_x, h_x, sampleSize);

  float *d_y = sycl::malloc_device<float>(samples, q);
  q.memcpy(d_y, h_y, sampleSize);

  float *d_z = sycl::malloc_device<float>(samples, q);
  q.memcpy(d_z, h_z, sampleSize);

  kdata *d_k = sycl::malloc_device<kdata>(CHUNK_S, q);

  const int ntpb = 256;
  const int nblks = (samples + ntpb - 1) / ntpb * ntpb;
  sycl::range<1> gws (nblks);
  sycl::range<1> lws (ntpb);

  int c = CHUNK_S;
  int s = sizeof(kdata) * c;
  int nchunks = (voxels + c - 1) / c;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nchunks; i++) {
    if (i == nchunks - 1) {
      c = voxels - CHUNK_S * i;
      s = sizeof(kdata) * c;
    }

    q.memcpy(d_k, &h_k[i * CHUNK_S], s);

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class fhd>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        cmpfhd( d_rmu + i*CHUNK_S,
                d_imu + i*CHUNK_S,
                d_rfhd, d_ifhd,
                d_x, d_y, d_z, d_k,
                samples, c, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device execution time %f (s)\n", time * 1e-9f);

  q.memcpy(rfhd, d_rfhd, sampleSize);
  q.memcpy(ifhd, d_ifhd, sampleSize);

  q.wait();

  if (verify) {
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
  }

  sycl::free(d_rmu, q);
  sycl::free(d_imu, q);
  sycl::free(d_rfhd, q);
  sycl::free(d_ifhd, q);
  sycl::free(d_x, q);
  sycl::free(d_y, q);
  sycl::free(d_z, q);
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
