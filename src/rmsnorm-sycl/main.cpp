#include "common.h"
#include "reference.h"
#include "reduce.h"
#include "utils.h"

template <typename T, int UNROLL, int THREADS_PER_WARP>
void rmsnorm_fwd_two_scan_kernel(const T *__restrict__ input,
                                 const T *__restrict__ gamma,
                                 T *__restrict__ output,
                                 const int64_t inner_len, const float epsilon,
                                 sycl::nd_item<3> &item,
                                 T *__restrict__ smem)
{
    const int BLOCKSIZE = item.get_local_range(2);
    const int bid = item.get_group(2);
    const int warp_id = item.get_local_id(2) / THREADS_PER_WARP;
    const int lane_id = item.get_local_id(2) % THREADS_PER_WARP;

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

    const float mean_square =
        BlockReduce<SumOp, float, THREADS_PER_WARP>(local_squares_sum, smem, item) /
        static_cast<float>(inner_len);
    const float norm_factor = sycl::rsqrt(mean_square + epsilon);

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

template <typename T, int THREADS_PER_WARP>
void rmsnorm_forward(sycl::queue &q,
                     const T *input, const T *gamma, T *output,
                     const int64_t inner_len, const int64_t outer_len,
                     const float epsilon, int block_size) {
    const sycl::range<3> lws (1, 1, block_size);
    const sycl::range<3> gws (1, 1, outer_len * block_size);
    constexpr int UNROLL = sizeof(sycl::uint4) / sizeof(T);
    if (inner_len % UNROLL == 0) {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<T, 1> smem(
            sycl::range<1>(MAX_THREADS_PER_BLOCK / THREADS_PER_WARP), cgh);
          cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
            rmsnorm_fwd_two_scan_kernel<T, UNROLL, THREADS_PER_WARP>(
              input, gamma, output, inner_len, epsilon, item,
              smem.template get_multi_ptr<sycl::access::decorated::no>().get());
          });
        });
    } else {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<T, 1> smem(
            sycl::range<1>(MAX_THREADS_PER_BLOCK / THREADS_PER_WARP), cgh);
          cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
            rmsnorm_fwd_two_scan_kernel<T, 1, THREADS_PER_WARP>(
              input, gamma, output, inner_len, epsilon, item,
              smem.template get_multi_ptr<sycl::access::decorated::no>().get());
          });
        });
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
  // create host memory of random numbers
  size_t size = (size_t)N * H;
  float* inp = make_random_float(size);
  float* gamma = make_random_float(H);
  float* out = (float*)malloc(size * sizeof(float));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int warpSize = *r;

  // move to GPU
  float* d_out;
  float* d_inp;
  float* d_gamma;
  d_out = sycl::malloc_device<float>(size, q);
  d_inp = sycl::malloc_device<float>(size, q);
  d_gamma = sycl::malloc_device<float>(H, q);
  q.memcpy(d_inp, inp, size * sizeof(float));
  q.memcpy(d_gamma, gamma, H * sizeof(float));
  q.wait();

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  rmsnorm_forward_cpu(out, inp, gamma, N, H);

  // check the correctness of the kernel at all block sizes
  for (int block_size : block_sizes) {
      printf("Checking block size %d.\n", block_size);

      if (warpSize == 64)
        rmsnorm_forward<float, 64>(q, d_inp, d_gamma, d_out, H, N, 1e-5f, block_size);
      else
        rmsnorm_forward<float, 32>(q, d_inp, d_gamma, d_out, H, N, 1e-5f, block_size);

      validate_result(q, d_out, out, "out", size, 1e-5f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  // time the kernel at different block sizes
  for (int block_size : block_sizes) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      if (warpSize == 64)
        rmsnorm_forward<float, 64>(q, d_inp, d_gamma, d_out, H, N, 1e-5f, block_size);
      else
        rmsnorm_forward<float, 32>(q, d_inp, d_gamma, d_out, H, N, 1e-5f, block_size);
    }

    q.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    float elapsed_time = duration.count();
    elapsed_time = elapsed_time / repeat;

    // estimate the memory bandwidth achieved
    long memory_ops = (2 * size + H) * 4; // *4 for float
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;
    printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
  }

  // free memory
  free(out);
  free(inp);
  free(gamma);
  sycl::free(d_out, q);
  sycl::free(d_inp, q);
  sycl::free(d_gamma, q);

  return 0;
}
