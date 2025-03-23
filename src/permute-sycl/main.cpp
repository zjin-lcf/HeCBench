/*
   Kernels for data permute
 */

#include <stdio.h>
#include <stdlib.h>
#include "common.hpp"

// CPU code reference
void permuate_cpu(float *inp, float *q, float *k, float *v, int B, int T, int C, int NH) {
  int i = 0;
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < NH; n++) {
      for (int t = 0; t < T; t++) {
        for (int c = n * (C/NH); c < (n+1)*(C/NH); c++) {
          q[i] = inp[b * T * 3 * C + t * 3 * C + c];
          k[i] = inp[b * T * 3 * C + t * 3 * C + C + c];
          v[i] = inp[b * T * 3 * C + t * 3 * C + 2*C + c];
          i++;
        }}}}
}

void permute_kernel(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    const float* inp,
    int B, int T, int NH, int d)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, T, d)
      // but instead, we have a single tensor QKV (inp) of shape (B, T, 3, NH, d)
      int idx = item.get_global_id(2);

      // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
      int C = NH * d;

      if (idx < B * C * T) {
        int b = idx / (C * T);
        int rest = idx % (C * T);
        int nh_ = rest / (T * d);
        rest = rest % (T * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
                (b * T * 3 * C)
                +   (n * 3 * C)
                +       (0 * C)
                +          (nh_ * d)
                +                d_;

        Q[idx] = inp[inp_idx];
        K[idx] = inp[inp_idx + C];
        V[idx] = inp[inp_idx + 2 * C];
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void permute (sycl::queue &q, float* out, const float* inp,
              int B, int T, int C, int NH,
              const int block_size)
{
  // inp is (B, T, 3C) QKV
  int HS = C / NH; // head size

  // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
  float *Q, *K, *V;
  Q = out + 0 * B * T * C;
  K = out + 1 * B * T * C;
  V = out + 2 * B * T * C;
  int total_threads = B * T * C;
  int num_blocks = ceil_div(total_threads, block_size);
  sycl::range<3> gws (1, 1, num_blocks * block_size);
  sycl::range<3> lws (1, 1, block_size);
  permute_kernel(q, gws, lws, 0, Q, K, V, inp, B, T, NH, HS);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int B = atoi(argv[1]);
  const int repeat_times = atoi(argv[2]);

  const int T = 1024;
  const int C = 768;
  const int NH = 12;

  size_t S = (size_t)B * T * C;

  // create host memory of random numbers
  float* inp = make_random_float(S * 3);
  float* out = make_random_float(S * 3);
  float* q = make_random_float(S);
  float* k = make_random_float(S);
  float* v = make_random_float(S);

  permuate_cpu(inp, q, k, v, B, T, C, NH);

  // move to GPU
#ifdef USE_GPU
  sycl::queue que(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue que(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_inp, *d_out;
  d_inp = sycl::malloc_device<float>(S * 3, que);
  que.memcpy(d_inp, inp, S * 3 * sizeof(float));
  d_out = sycl::malloc_device<float>(S * 3, que);

  int block_sizes[] = {32, 64, 128, 256, 512};

  for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);
    permute (que, d_out, d_inp, B, T, C, NH, block_size);
    validate_result(d_out, q, "q", S, 1e-6f);
    validate_result(d_out+B*T*C, k, "k", S, 1e-6f);
    validate_result(d_out+2*B*T*C, v, "v", S, 1e-6f);
  }
  printf("All results match. Starting benchmarks.\n\n");

  // benchmark speed of the kernel
  for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    float elapsed_time = benchmark_kernel(repeat_times, permute,
        que, d_out, d_inp, B, T, C, NH, block_size);

    printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
  }

  // free memory
  free(inp);
  free(q);
  free(k);
  free(v);
  free(out);
  sycl::free(d_inp, que);
  sycl::free(d_out, que);

  return 0;
}
