/*
   Kernels for data permute 
 */

#include <stdio.h>
#include <stdlib.h>
#include "common.h"

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

void permute (float* out, const float* inp,
    int B, int T, int C, int NH,
    const int block_size) {
  // inp is (B, T, 3C) QKV
  int d = C / NH; // head size (HS)

  // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
  float *q, *k, *v;
  q = out + 0 * B * T * C;
  k = out + 1 * B * T * C;
  v = out + 2 * B * T * C;
  int total_threads = B * T * C;
  int num_blocks = ceil_div(total_threads, block_size);
  //permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
  #pragma omp target teams distribute parallel for \
   num_teams(num_blocks) num_threads(block_size) 
  for (int idx = 0; idx < B * T * C; idx++) { 
    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
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

    q[idx] = inp[inp_idx];
    k[idx] = inp[inp_idx + C];
    v[idx] = inp[inp_idx + 2 * C];
  }
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

  int block_sizes[] = {32, 64, 128, 256, 512};

  #pragma omp target data map(to: inp[0:S*3]) map(alloc: out[0:S*3])
  {
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
      int block_size = block_sizes[j];
      printf("Checking block size %d.\n", block_size);
      permute (out, inp, B, T, C, NH, block_size);
      validate_result(out, q, "q", S, 1e-6f);
      validate_result(out+B*T*C, k, "k", S, 1e-6f);
      validate_result(out+2*B*T*C, v, "v", S, 1e-6f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
      int block_size = block_sizes[j];
      float elapsed_time = benchmark_kernel(repeat_times, permute,
          out, inp, B, T, C, NH, block_size);

      printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }
  }

  // free memory
  free(inp);
  free(q);
  free(k);
  free(v);
  free(out);

  return 0;
}
