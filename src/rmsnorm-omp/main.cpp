#include "common.h"
#include "reference.h"

void rmsnorm_forward_kernel (const float* __restrict inp, const float* __restrict gamma, float* out, 
                             int64_t N, int64_t H, float eps, int block_size)
{
  #pragma omp target teams distribute num_teams(N)
  for (int64_t t = 0; t < N; t++) {
    const float* x = inp + t * H;
    
    // RMS
    float m = 0.0f;
    #pragma omp parallel for reduction(+:m) num_threads(block_size)
    for (int i = 0; i < H; i++) {
    	m += x[i] * x[i];
    }
    m = m/H;
    float s = 1.0f / sqrtf(m + eps);
    
    float* out_t = out + t * H;
    #pragma omp parallel for num_threads(block_size)
    for (int64_t i = 0; i < H; i++) {
      float o = x[i] * s * gamma[i];
      out_t[i] = o;
    }
  }
}

int main(int argc, char **argv) {

  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  srand(0);

  size_t size = (size_t)N * H;

  // create host memory of random numbers
  float* inp = make_random_float(size);
  float* gamma = make_random_float(H);

  float* out = (float*)malloc(size * sizeof(float));
  float* d_out = (float*)malloc(size * sizeof(float));

  // move to GPU
  #pragma omp target data map(to: inp[0:size], gamma[0:H]) \
                          map(alloc: d_out[0:size])
  {
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    rmsnorm_forward_cpu(out, inp, gamma, N, H);

    // check the correctness of the kernel at all block sizes
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        rmsnorm_forward_kernel(inp, gamma, d_out, N, H, 1e-5f, block_size);

        validate_result(d_out, out, "out", size, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (size_t j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
      int block_size = block_sizes[j];

      auto start = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < repeat; i++) {
        rmsnorm_forward_kernel(inp, gamma, d_out, N, H, 1e-5f, block_size);
      }

      auto stop = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration = stop - start;
      float elapsed_time = duration.count();
      elapsed_time = elapsed_time / repeat;

      // estimate the memory bandwidth achieved
      size_t memory_ops = (2 * size + H) * 4; // *4 for float
      float memory_bandwidth = memory_ops / elapsed_time / 1e6;
      printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }
  }

  // free memory
  free(out);
  free(d_out);
  free(inp);
  free(gamma);

  return 0;
}
