#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "common.h"

#define TILE_SIZE 5900
#define NTHREADS 256

// 1,2,3,4,5,6 -> 2,3,4,6,1,5
static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int repeat = 1;

static const int shape_output[] = {d2, d3, d1};
static const int shape_input[] = {d4, d5, d6};
static const float shape_output_r[] = {1.0 / d2, 1.0 / d3, 1.0 / d1};
static const float shape_input_r[] = {1.0 / d4, 1.0 / d5, 1.0 / d6};
static const int stride_output_local[] = {d1, d1 * d2, 1};
static const int stride_output_global[] = {1, d2, d2 * d3 * d4 * d6};
static const int stride_input[] = {d2 * d3, d2 * d3 * d4 * d6 * d1, d2 * d3 * d4};

void verify(double *input, double *output) {
  int input_offset  = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  bool error = false;
  for (size_t i = 0; i < d5; i++) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] != 
        output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      printf("FAIL\n");
      error = true;
      break;
    }
  }
  if (!error) printf("PASS\n");
}

int main(int argv, char **argc) {
  if (argv > 1) {
    repeat = atoi(argc[1]);
  }

  double *input = new double[data_size]();
  double *output = new double[data_size]();

  for (size_t i = 0; i < data_size; i++) {
    input[i] = i;
  }

  const int nblocks = d4 * d5 * d6;
  const int tile_size = d1 * d2 * d3;
  const int dim_output = 3;
  const int dim_input = 3;

  {

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    buffer<double, 1> d_output(output, data_size);
    buffer<double, 1> d_input(input, data_size);
    buffer<int, 1> d_shape_input(shape_input, dim_input);
    buffer<float, 1> d_shape_input_r(shape_input_r, dim_input);
    buffer<int, 1> d_shape_output(shape_output, dim_output);
    buffer<float, 1> d_shape_output_r(shape_output_r, dim_output);
    buffer<int, 1> d_stride_input(stride_input, dim_input);
    buffer<int, 1> d_stride_output_local(stride_output_local, dim_output);
    buffer<int, 1> d_stride_output_global(stride_output_global, dim_output);

    range<1> gws (nblocks * NTHREADS);
    range<1> lws (NTHREADS);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < repeat; ++i) {
      q.submit([&] (handler &cgh) {
        auto output = d_output.get_access<sycl_discard_write>(cgh);
        auto input = d_input.get_access<sycl_read>(cgh);
        auto shape_input = d_shape_input.get_access<sycl_read>(cgh);
        auto shape_input_r = d_shape_input_r.get_access<sycl_read>(cgh);
        auto shape_output = d_shape_output.get_access<sycl_read>(cgh);
        auto shape_output_r = d_shape_output_r.get_access<sycl_read>(cgh);
        auto stride_input = d_stride_input.get_access<sycl_read>(cgh);
        auto stride_output_local = d_stride_output_local.get_access<sycl_read>(cgh);
        auto stride_output_global = d_stride_output_global.get_access<sycl_read>(cgh);

        accessor<double, 1, sycl_read_write, access::target::local> tile(TILE_SIZE, cgh);
        cgh.parallel_for<class transpose>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          for (int block_idx = item.get_group(0); block_idx < nblocks;
                   block_idx += item.get_group_range(0)) {
            int it = block_idx, im = 0, offset1 = 0;
            for (int i = 0; i < dim_input; i++) {
              im = it * shape_input_r[i];
              offset1 += stride_input[i] * (it - im * shape_input[i]);
              it = im;
            }

            for (int i = item.get_local_id(0); i < tile_size; i += item.get_local_range(0)) {
              tile[i] = input[i + block_idx * tile_size];
            }

            item.barrier(access::fence_space::local_space);

            for (int i = item.get_local_id(0); i < tile_size; i += item.get_local_range(0)) {
              it = i;
              int offset2 = 0, local_offset = 0;
              for (int j = 0; j < dim_output; j++) {
                im = it * shape_output_r[j];
                int tmp = it - im * shape_output[j];
                offset2 += stride_output_global[j] * tmp;
                local_offset += stride_output_local[j] * tmp;
                it = im;
              }
              output[offset1 + offset2] = tile[local_offset];
            }
            item.barrier(access::fence_space::local_space);
          }
        });
      });
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  verify(input, output);
  delete [] input;
  delete [] output;
  return 0;
}
