#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "kernels.h"
#include "reference.h"

void zoom (int repeat, int input_sizes[4], float zoom_factor[2])
{
  int N = input_sizes[0];
  int C = input_sizes[1];
  int H = input_sizes[2];
  int W = input_sizes[3];

  int Ho = (int)floorf(H * zoom_factor[0]);
  int Wo = (int)floorf(W * zoom_factor[1]);

  bool is_zoom_out = Ho < H && Wo < W;
  bool is_zoom_in = Ho > H && Wo > W;
  if (is_zoom_out == false && is_zoom_in == false) {
    printf("Zoom factors only handle simultaneous expansion(or shrinkage) in both dimensions. Exit\n");
    exit(1);
  }

  // image pitch
  size_t pitch = (size_t)H * W;

  int pad_dims[2][2] = {{0, 0}, {0,0}};  // zoom out
  int slice_dims[2][2] = {{0, 0}, {0,0}};  // zoom in

  int diff = H - Ho;
  int half = abs(diff) / 2;
  if (diff > 0) {
    pad_dims[0][0] = half;
    pad_dims[0][1] = diff - half;
  } else {
    slice_dims[0][0] = half;
    slice_dims[0][1] = H + half;
  }

  diff = W - Wo;
  half = abs(diff) / 2;
  if (diff > 0) {
    pad_dims[1][0] = half;
    pad_dims[1][1] = diff - half;
  } else {
    slice_dims[1][0] = half;
    slice_dims[1][1] = W + half;
  }

  size_t img_size = pitch * N * C;
  size_t img_size_bytes = sizeof(float) * img_size;

  size_t output_img_size = img_size;
  size_t output_img_size_bytes = sizeof(float) * output_img_size;

  float *input_img = (float*) malloc (img_size_bytes);

  float *output_img = (float*) malloc (output_img_size_bytes);
  float *output_img_ref = (float*) calloc (output_img_size, sizeof(float));

  std::default_random_engine rng (123);
  std::normal_distribution<float> norm_dist(0.f, 1.f);

  for (size_t i = 0; i < img_size; i++) {
    input_img[i] = norm_dist(rng);
  }

  #pragma omp target data map(to: input_img[0:img_size]) \
                          map(from: output_img[0:output_img_size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {

      if (is_zoom_in) {
        zoom_in_kernel(input_img, output_img, H, W, Ho, Wo,
                       pitch, slice_dims[0][0],
                       slice_dims[0][1],
                       slice_dims[1][0],
                       slice_dims[1][1],
                       C * N);
      }
      else if (is_zoom_out) {
        zoom_out_kernel(input_img, output_img, H, W, Ho, Wo,
                        pitch, pad_dims[0][0],
                        pad_dims[0][1],
                        pad_dims[1][0],
                        pad_dims[1][1],
                        C * N);

        zoom_out_edge_pad(output_img, H, W, pitch,
                          pad_dims[0][0], pad_dims[1][0],
                          pad_dims[0][0] + Ho,
                          pad_dims[1][0] + Wo,
                          C * N);
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the %s kernel: %f (us)\n",
           is_zoom_in ? "zoom-in" : "zoom-out", time * 1e-3 / repeat);
  }

  if (is_zoom_in) {
    zoom_in_reference(
        input_img, output_img_ref, H, W, Ho, Wo,
        pitch, slice_dims[0][0],
        slice_dims[0][1],
        slice_dims[1][0],
        slice_dims[1][1],
        C * N);
  } else if (is_zoom_out) {
    zoom_out_reference(
      input_img, output_img_ref, H, W, Ho, Wo,
      pitch, pad_dims[0][0],
      pad_dims[0][1],
      pad_dims[1][0],
      pad_dims[1][1],
      C * N);

    zoom_out_edge_pad_reference(
      output_img_ref, H, W, pitch,
      pad_dims[0][0], pad_dims[1][0],
      pad_dims[0][0] + Ho,
      pad_dims[1][0] + Wo,
      C * N);
  }

  bool ok = true;
  for (size_t i = 0; i < output_img_size; i++) {
    if (fabsf(output_img[i] - output_img_ref[i]) > 1e-4f) {
      ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input_img);
  free(output_img);
  free(output_img_ref);
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <batch> <channel> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_sizes[4];
  input_sizes[0] = atoi(argv[1]);
  input_sizes[1] = atoi(argv[2]);
  input_sizes[2] = atoi(argv[3]);
  input_sizes[3] = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  float zf[2]; // zoom factor

  zf[0] = 1.5f; zf[1] = 2.5f;
  zoom(repeat, input_sizes, zf);

  zf[0] = 0.6f; zf[1] = 0.9f;
  zoom(repeat, input_sizes, zf);

  return 0;
}
