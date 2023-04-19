#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>

#define BLOCK_SIZE 256

void findMovingPixels(
  sycl::nd_item<1> &item,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Img1,
  const unsigned char *__restrict Img2,
  const unsigned char *__restrict Tn,
        unsigned char *__restrict Mp) // moving pixel map
{
  size_t i = item.get_global_id(0);
  if (i >= imgSize) return;
  if ( sycl::abs(Img[i] - Img1[i]) > Tn[i] || sycl::abs(Img[i] - Img2[i]) > Tn[i] )
    Mp[i] = 255;
  else {
    Mp[i] = 0;
  }
}

// alpha = 0.92 
void updateBackground(
  sycl::nd_item<1> &item,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Mp,
        unsigned char *__restrict Bn)
{
  size_t i = item.get_global_id(0);
  if (i >= imgSize) return;
  if ( Mp[i] == 0 ) Bn[i] = 0.92f * Bn[i] + 0.08f * Img[i];
}

// alpha = 0.92, c = 3
void updateThreshold(
  sycl::nd_item<1> &item,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Mp,
  const unsigned char *__restrict Bn,
        unsigned char *__restrict Tn)
{
  size_t i = item.get_global_id(0);
  if (i >= imgSize) return;
  if (Mp[i] == 0) {
    float th = 0.92f * Tn[i] + 0.24f * (Img[i] - Bn[i]);
    Tn[i] = sycl::fmax(th, 20.f);
  }
}

//
// merge three kernels into a single kernel
//
void merge(
  sycl::nd_item<1> &item,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Img1,
  const unsigned char *__restrict Img2,
        unsigned char *__restrict Tn,
        unsigned char *__restrict Bn)
{
  size_t i = item.get_global_id(0);
  if (i >= imgSize) return;
  if ( sycl::abs(Img[i] - Img1[i]) <= Tn[i] && sycl::abs(Img[i] - Img2[i]) <= Tn[i] ) {
    // update background
    Bn[i] = 0.92f * Bn[i] + 0.08f * Img[i];

    // update threshold
    float th = 0.92f * Tn[i] + 0.24f * (Img[i] - Bn[i]);
    Tn[i] = sycl::fmax(th, 20.f);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <merge> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int merged = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int imgSize = width * height;
  const size_t imgSize_bytes = imgSize * sizeof(char);
  unsigned char *Img = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Bn = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Tn = (unsigned char*) malloc (imgSize_bytes);

  std::mt19937 generator (123);
  std::uniform_int_distribution<int> distribute( 0, 255 );

  for (int j = 0; j < imgSize; j++) {
    Bn[j] = distribute(generator);
    Tn[j] = 128;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned char *d_Img, *d_Img1, *d_Img2;
  unsigned char *d_Bn, *d_Mp, *d_Tn;
  d_Img = sycl::malloc_device<unsigned char>(imgSize, q);
  d_Img1 = sycl::malloc_device<unsigned char>(imgSize, q);
  d_Img2 = sycl::malloc_device<unsigned char>(imgSize, q);
  d_Bn = sycl::malloc_device<unsigned char>(imgSize, q);
  d_Mp = sycl::malloc_device<unsigned char>(imgSize, q);
  d_Tn = sycl::malloc_device<unsigned char>(imgSize, q);

  q.memcpy(d_Bn, Bn, imgSize_bytes);
  q.memcpy(d_Tn, Tn, imgSize_bytes);

  sycl::range<1> gws ((imgSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  long time = 0;

  for (int i = 0; i < repeat; i++) {

    for (int j = 0; j < imgSize; j++) {
      Img[j] = distribute(generator);
    }

    q.memcpy(d_Img, Img, imgSize_bytes).wait();

    // Time t   : Image   | Image1   | Image2
    // Time t+1 : Image2  | Image    | Image1
    // Time t+2 : Image1  | Image2   | Image
    unsigned char *t = d_Img2;
    d_Img2 = d_Img1;
    d_Img1 = d_Img;
    d_Img = t;

    if (i >= 2) {
      if (merged) {
        auto start = std::chrono::steady_clock::now();
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<class merged_kernel>(
            sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            merge ( item, imgSize, d_Img, d_Img1, d_Img2, d_Tn, d_Bn );
          });
        }).wait();
        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      else {
        auto start = std::chrono::steady_clock::now();
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<class k1>(
            sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            findMovingPixels ( item, imgSize, d_Img, d_Img1, d_Img2, d_Tn, d_Mp );
          });
        });
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<class k2>(
            sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            updateBackground  ( item, imgSize, d_Img, d_Mp, d_Bn );
          });
        });
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<class k3>(
            sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            updateThreshold  ( item, imgSize, d_Img, d_Mp, d_Bn, d_Tn );
          });
        });
        q.wait();
        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
    }
  }

  float kernel_time = (repeat <= 2) ? 0 : (time * 1e-3f) / (repeat - 2);
  printf("Average kernel execution time: %f (us)\n", kernel_time);

  q.memcpy(Tn, d_Tn, imgSize_bytes).wait();

  // verification
  int sum = 0;
  int bin[4] = {0, 0, 0, 0};
  for (int j = 0; j < imgSize; j++) {
    sum += abs(Tn[j] - 128);
    if (Tn[j] < 64)
      bin[0]++;
    else if (Tn[j] < 128)
      bin[1]++;
    else if (Tn[j] < 192)
      bin[2]++;
    else
      bin[3]++;
  }
  sum = sum / imgSize;
  printf("Average threshold change is %d\n", sum);
  printf("Bin counts are %d %d %d %d\n", bin[0], bin[1], bin[2], bin[3]);
     
  free(Img);
  free(Tn);
  free(Bn);
  sycl::free(d_Img, q);
  sycl::free(d_Img1, q);
  sycl::free(d_Img2, q);
  sycl::free(d_Tn, q);
  sycl::free(d_Mp, q);
  sycl::free(d_Bn, q);

  return 0;
}
