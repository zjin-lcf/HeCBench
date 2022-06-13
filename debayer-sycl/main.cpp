#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"
#include "util.h"
#include "image.h"
#include "kernel.h"

int main(int argc, char* argv[]) 
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_image_pitch = width;
  const int output_image_pitch = width * 4;
  const int numPix = width * height;
  const int input_image_size = numPix * sizeof(uchar);
  const int output_image_size = numPix * 4 * sizeof(uchar);

  uchar *input = (uchar*) malloc (input_image_size);
  uchar *output = (uchar*) malloc (output_image_size);

  // RGGB, GRBG, GBRG, BGGR
  const int bayer_pattern = RGGB;

  // input image with random values
  srand(123);
  for (int i = 0; i < numPix; i++) {
    input[i] = rand() % 256;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);
  buffer<uchar, 1> d_input (input, numPix);
  buffer<uchar, 1> d_output (numPix);

  range<2> gws ((height + tile_rows - 1) / tile_rows * tile_rows, 
                (width + tile_cols - 1) / tile_cols * tile_cols);
  range<2> lws (tile_rows, tile_cols);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  //this version takes a tile (z=1) and each tile job does 4 line median sorts
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto om = d_output.get_access<sycl_discard_write>(cgh);
      cgh.fill(om, (uchar)0);
    });

    q.submit([&] (handler &cgh) {
      auto im = d_input.get_access<sycl_read>(cgh);
      auto om = d_output.get_access<sycl_read_write>(cgh);
      accessor<LDSPixelT, 1, sycl_read_write, access::target::local> apron(apron_rows * apron_cols, cgh);
      cgh.parallel_for<class debayer>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        malvar_he_cutler_demosaic (
          item, apron.get_pointer(), height, width, 
          im.get_pointer(), input_image_pitch,
          om.get_pointer(), output_image_pitch,
          bayer_pattern );
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  q.submit([&] (handler &cgh) {
    auto acc = d_output.get_access<sycl_read>(cgh);
    cgh.copy(acc, output);
  }).wait();

  long sum = 0;
  for (int i = 0; i < numPix; i++) sum += output[i];
  printf("Checksum: %ld\n", sum);

  free(input);
  free(output);
  return 0;
}
