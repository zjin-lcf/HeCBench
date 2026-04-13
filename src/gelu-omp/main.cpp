#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "reference.h"

using __half = _Float16;

struct alignas(4) __half2 {
  __half x;
  __half y;
};

// width is hidden_dim and height is seq_len
void gelu_bias_loop(__half* src, const __half* bias, int batch_size, int width, int height, int block_size)
{
  #pragma omp target teams distribute collapse(2) num_teams(batch_size * height)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int x = 0; x < height; x++) {
      int index = (batch * width * height + x * width) >> 1;
      #pragma omp parallel for num_threads(block_size)
      for (int y = 0; y < width / 2; y++) {
        auto v_bias = ((__half2*)bias)[y];
        auto v_src  = ((__half2*)src)[index + y];
        auto tx      = float(v_src.x + v_bias.x);
        auto ty      = float(v_src.y + v_bias.y);
        tx    = 0.5f * tx * (1.0f + tanhf(0.79788456f * (tx + 0.044715f * tx * tx * tx)));
        ty    = 0.5f * ty * (1.0f + tanhf(0.79788456f * (ty + 0.044715f * ty * ty * ty)));
        ((__half2*)src)[index + y] = {__half(tx), __half(ty)};
      }
    }
  }
}

void gelu_bias_loop_base(__half* src, const __half* bias, int batch_size, int width, int height, int block_size)
{
  #pragma omp target teams distribute collapse(2) num_teams(batch_size * height)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int x = 0; x < height; x++) {
      int index = batch * width * height + x * width;
      #pragma omp parallel for num_threads(block_size)
      for (int y = 0; y < width; y++) {
        auto v_bias = bias[y];
        auto v_src  = src[index + y];
        auto v      = v_src + v_bias;
        auto t      = float(v);
        t = 0.5f * t * (1.0f + tanhf(0.79788456f * (t + 0.044715f * t * t * t)));
        src[index + y] = __half(t);
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <batch> <sequence length> <hidden dimension> <repeat>\n", argv[0]);
    printf("The hidden dimension is a multiple of two\n");
    return 1;
  }

  const int batch_size = atoi(argv[1]);
  const int seq_len = atoi(argv[2]);
  const int hidden_dim = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const size_t src_size = (size_t)batch_size * seq_len * hidden_dim;

  const size_t src_size_bytes =  src_size * sizeof(__half);
  const int bias_size_bytes = hidden_dim * sizeof(__half);

  srand(123);
  __half* output = (__half*) malloc (src_size_bytes);
  __half* output_ref = (__half*) malloc (src_size_bytes);
  for (size_t i = 0; i < src_size; i++) {
    output_ref[i] = output[i] = __half(rand() / (float)RAND_MAX); // input and output
  }

  __half* bias = (__half*) malloc (bias_size_bytes);
  for (int i = 0; i < hidden_dim; i++) {
    bias[i] = __half(-6 + (rand() % 12));
  }

  int block_size;
  if (hidden_dim >= 4096)
    block_size = 512;
  else if (hidden_dim >= 2048)
    block_size = 256;
  else
    block_size = 128;

  // warmup and verify
  gelu_bias_loop_cpu (output_ref, bias, batch_size, hidden_dim, seq_len);

  #pragma omp target data map(to: bias[0:hidden_dim], output[0:src_size])
  {
    //gelu_bias_loop_base(output, bias, batch_size, hidden_dim, seq_len, block_size);
    gelu_bias_loop(output, bias, batch_size, hidden_dim, seq_len, block_size);

    #pragma omp target update from (output[0:src_size])

    bool ok = true;
    for (size_t i = 0; i < src_size; i++) {
      if (fabsf(float(output_ref[i]) - float(output[i])) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      gelu_bias_loop(output, bias, batch_size, hidden_dim, seq_len, block_size);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of vectorized kernel %f (ms)\n", (time * 1e-6f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      gelu_bias_loop_base(output, bias, batch_size, hidden_dim, seq_len, block_size);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of baseline kernel %f (ms)\n", (time * 1e-6f) / repeat);
  }

  free(output);
  free(output_ref);
  free(bias);

  return 0;
}
