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
  const int imageSize_bytes = imageSize * sizeof(uchar3);

  uchar3* h_src = (uchar3*) malloc (imageSize_bytes);
  uchar3* h_dst = (uchar3*) malloc (imageSize_bytes);
  uchar3* r_dst = (uchar3*) malloc (imageSize_bytes);

  srand(123);
  for (int i = 0; i < imageSize; i++) {
    h_src[i] = {rand() % 256, rand() % 256, rand() % 256};
  }

  {
  #ifdef USE_GPU
    gpu_selector dev_sel;
  #else
    cpu_selector dev_sel;
  #endif
    queue q(dev_sel);
  
    buffer<uchar3, 1> d_src(h_src, imageSize);
    buffer<uchar3, 1> d_dst(h_dst, imageSize);
    buffer<Properties, 1> d_prop (&prop, 1);
  
    range<2> lws (16, 16);
    range<2> gws ((height / 16 + 1) * 16, (width / 16 + 1) * 16);
  
    q.wait();
    auto start = std::chrono::steady_clock::now();
  
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (handler &cgh) {
        auto src = d_src.get_access<sycl_read>(cgh);
        auto dst = d_dst.get_access<sycl_discard_write>(cgh);
        auto prop = d_prop.get_access<sycl_read>(cgh);
        cgh.parallel_for<class scan_block>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          barrel_distort(item,
                         src.get_pointer(),
                         dst.get_pointer(),
                         prop.get_pointer());
        });
      });
    }
  
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
  }

  // verify
  int ex = 0, ey = 0, ez = 0;
  reference(h_src, r_dst, &prop);
  for (int i = 0; i < imageSize; i++) {
    ex = max(abs(h_dst[i].x() - r_dst[i].x()), ex);
    ey = max(abs(h_dst[i].y() - r_dst[i].y()), ey);
    ez = max(abs(h_dst[i].z() - r_dst[i].z()), ez);
  }

  std::cout << "Max error of each channel: " << ex << " " << ey << " " << ez << std::endl;

  free(h_src);
  free(h_dst);

  return 0;
}
