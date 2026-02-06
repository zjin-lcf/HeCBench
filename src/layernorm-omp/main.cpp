/*
Kernels for layernorm forward pass.
*/

#include "common.h"
#include "reference.h"

void layernorm_forward_kernel(float*__restrict out,
                              float*__restrict mean,
                              float*__restrict rstd,
                              const float*__restrict inp,
                              const float*__restrict weight,
                              const float*__restrict bias,
                              int B, int T, int C,
                              int block_size)
{
    #pragma omp target teams distribute collapse(2) num_teams(B*T)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            #pragma omp parallel for reduction(+:m) num_threads(block_size)
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            #pragma omp parallel for reduction(+:v) num_threads(block_size)
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + 1e-5f);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            #pragma omp parallel for num_threads(block_size)
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}


int main(int argc, char **argv) {

  if (argc != 5) {
    printf("Usage: %s <batch size> <sequence length> <channel length> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t B = atoi(argv[1]);
  const size_t T = atoi(argv[2]);
  const size_t C = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  // create host memory of random numbers
  srand(0);
  float* out = (float*)malloc(B * T * C * sizeof(float));
  float* d_out = (float*)malloc(B * T * C * sizeof(float));
  float* mean = (float*)malloc(B * T * sizeof(float));
  float* d_mean = (float*)malloc(B * T * sizeof(float));
  float* rstd = (float*)malloc(B * T * sizeof(float));
  float* d_rstd = (float*)malloc(B * T * sizeof(float));
  float* inp = make_random_float(B * T * C);
  float* weight = make_random_float(C);
  float* bias = make_random_float(C);

  // move to GPU
  #pragma omp target data map(to: inp[0:B*T*C], weight[0:C], bias[0:C]) \
                          map(alloc: d_out[0:B*T*C], d_mean[0:B*T], d_rstd[0:B*T])
  {

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int block_size : block_sizes) {
        printf("Checking block size %d.\n", block_size);

        layernorm_forward_kernel(d_out, d_mean, d_rstd, inp, weight, bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int block_size : block_sizes) {

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < repeat; i++) {
          layernorm_forward_kernel(d_out, d_mean, d_rstd, inp, weight, bias, B, T, C, block_size);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = stop - start;
        float elapsed_time = duration.count();
        elapsed_time = elapsed_time / repeat;

        // estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }
  }

  // free memory
  free(out);
  free(d_out);
  free(mean);
  free(d_mean);
  free(rstd);
  free(d_rstd);
  free(inp);
  free(weight);
  free(bias);

  return 0;
}
