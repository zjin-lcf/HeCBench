#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "distort.h"

int main(int argc, char **argv)
{
  if (argc != 5) {
    std::cout << "Usage: " << argv[0] <<
      "<input image width> <input image height> <coefficient of distortion> <repeat>\n";
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const float K = atof(argv[3]);
  const int repeat = atoi(argv[4]);

  struct Properties prop;
  prop.K = K;
  prop.centerX = width / 2;
  prop.centerY = height / 2;
  prop.width = width;
  prop.height = height;
  prop.thresh = 1.f;

  prop.xshift = calc_shift(0, prop.centerX - 1, prop.centerX, prop.K, prop.thresh);
  float newcenterX = prop.width - prop.centerX;
  float xshift_2 = calc_shift(0, newcenterX - 1, newcenterX, prop.K, prop.thresh);

  prop.yshift = calc_shift(0, prop.centerY - 1, prop.centerY, prop.K, prop.thresh);
  float newcenterY = prop.height - prop.centerY;
  float yshift_2 = calc_shift(0, newcenterY - 1, newcenterY, prop.K, prop.thresh);

  prop.xscale = (prop.width - prop.xshift - xshift_2) / prop.width;
  prop.yscale = (prop.height - prop.yshift - yshift_2) / prop.height;

  const int imageSize = height * width;
  const size_t imageSize_bytes = imageSize * sizeof(sycl::uchar3);

  sycl::uchar3* h_src = (sycl::uchar3*) malloc (imageSize_bytes);
  sycl::uchar3* h_dst = (sycl::uchar3*) malloc (imageSize_bytes);
  sycl::uchar3* r_dst = (sycl::uchar3*) malloc (imageSize_bytes);

  srand(123);
  for (int i = 0; i < imageSize; i++) {
    h_src[i] = {rand() % 256, rand() % 256, rand() % 256};
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  sycl::uchar3 *d_src = sycl::malloc_device<sycl::uchar3>(imageSize, q);
  q.memcpy(d_src, h_src, imageSize_bytes);

  sycl::uchar3 *d_dst = sycl::malloc_device<sycl::uchar3>(imageSize, q);

  Properties *d_prop = sycl::malloc_device<Properties>(1, q);
  q.memcpy(d_prop, &prop, sizeof(Properties));
  
  sycl::range<2> lws (16, 16);
  sycl::range<2> gws ((height / 16 + 1) * 16, (width / 16 + 1) * 16);
  
  q.wait();
  auto start = std::chrono::steady_clock::now();
  
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class image_process>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        barrel_distort(item, d_src, d_dst, d_prop);
      });
    });
  }
  
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(h_dst, d_dst, imageSize_bytes).wait();

  // verify
  int ex = 0, ey = 0, ez = 0;
  reference(h_src, r_dst, &prop);
  for (int i = 0; i < imageSize; i++) {
    ex = std::max(abs(h_dst[i].x() - r_dst[i].x()), ex);
    ey = std::max(abs(h_dst[i].y() - r_dst[i].y()), ey);
    ez = std::max(abs(h_dst[i].z() - r_dst[i].z()), ez);
  }

  std::cout << "Max error of each channel: " << ex << " " << ey << " " << ez << std::endl;

  sycl::free(d_src, q);
  sycl::free(d_dst, q);
  sycl::free(d_prop, q);

  free(h_src);
  free(h_dst);

  return 0;
}
