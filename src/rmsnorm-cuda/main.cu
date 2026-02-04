#include "common.h"
#include "reference.h"
#include "reduce.cuh"
#include "utils.cuh"

template <typename T, int UNROLL>
__global__
void rmsnorm_fwd_two_scan_kernel(const T *__restrict__ input,
                                 const T *__restrict__ gamma,
                                       T *output,
                                 const int64_t inner_len, const float epsilon)
{
    const int BLOCKSIZE = blockDim.x;
    const int bid       = blockIdx.x;
    const int warp_id   = threadIdx.x / THREADS_PER_WARP;
    const int lane_id   = threadIdx.x % THREADS_PER_WARP;

    const T *input_ptr  = input + bid * inner_len;
    const T *gamma_ptr  = gamma;
    T       *output_ptr = output + bid * inner_len;

    const int start_offset = warp_id * THREADS_PER_WARP * UNROLL + lane_id * UNROLL;
    T         ld_input_regs[UNROLL];
    float     local_squares_sum = 0.0f;
    for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCKSIZE * UNROLL)) {
        load_data<T, UNROLL>(input_ptr + offset, ld_input_regs);
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            const float val = static_cast<float>(ld_input_regs[i]);
            local_squares_sum += (val * val);
        }
    }

    const float mean_square = BlockReduce<SumOp, float>(local_squares_sum) / static_cast<float>(inner_len);
    const float norm_factor = rsqrtf(mean_square + epsilon);

    T ld_gamma_regs[UNROLL];
    T st_regs[UNROLL];
    for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCKSIZE * UNROLL)) {
        load_data<T, UNROLL>(input_ptr + offset, ld_input_regs);
        load_data<T, UNROLL>(gamma_ptr + offset, ld_gamma_regs);

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            float val = static_cast<float>(ld_input_regs[i]) * norm_factor *
                        static_cast<float>(ld_gamma_regs[i]);
            st_regs[i] = static_cast<T>(val);
        }
        store_data<T, UNROLL>(output_ptr + offset, st_regs);
    }
}

template <typename T>
void rmsnorm_forward(const T *input, const T *gamma, T *output, const int64_t inner_len,
                      const int64_t outer_len, const float epsilon, int block_size) {
    const dim3    block_dim(block_size, 1, 1);
    const dim3    grid_dim(outer_len, 1, 1);
    constexpr int UNROLL = sizeof(uint4) / sizeof(T);
    if (inner_len % UNROLL == 0) {
        rmsnorm_fwd_two_scan_kernel<T, UNROLL>
            <<<grid_dim, block_dim>>>(input, gamma, output, inner_len, epsilon);
    } else {
        rmsnorm_fwd_two_scan_kernel<T, 1>
            <<<grid_dim, block_dim>>>(input, gamma, output, inner_len, epsilon);
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

  // move to GPU
  float* d_out;
  float* d_inp;
  float* d_gamma;
  cudaCheck(cudaMalloc(&d_out, size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_inp, size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_gamma, H * sizeof(float)));
  cudaCheck(cudaMemcpy(d_inp, inp, size * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_gamma, gamma, H * sizeof(float), cudaMemcpyHostToDevice));

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  rmsnorm_forward_cpu(out, inp, gamma, N, H);

  // check the correctness of the kernel at all block sizes
  for (int block_size : block_sizes) {
      printf("Checking block size %d.\n", block_size);

      rmsnorm_forward(d_inp, d_gamma, d_out, H, N, 1e-5f, block_size);

      validate_result(d_out, out, "out", size, 1e-5f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  // time the kernel at different block sizes
  for (int block_size : block_sizes) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      rmsnorm_forward(d_inp, d_gamma, d_out, H, N, 1e-5f, block_size);
    }

    cudaCheck(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    float elapsed_time = duration.count();
    elapsed_time = elapsed_time / repeat;

    // estimate the memory bandwidth achieved
    size_t memory_ops = (2 * size + H) * 4; // *4 for float
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;
    printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
  }

  // free memory
  free(out);
  free(inp);
  free(gamma);
  cudaCheck(cudaFree(d_out));
  cudaCheck(cudaFree(d_inp));
  cudaCheck(cudaFree(d_gamma));

  return 0;
}
