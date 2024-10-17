/*
   Kernels for softmax forward pass.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include "common.h"
#include "reference.h"

// online softmax paper: http://arxiv.org/abs/1805.02867
// online softmax reduces loops from 3 to 2
// which is done by calculating sumval and maxval in one loop

constexpr int warp_size = 32;

/*
// struct for the reduction operation, guarantees 8-byte alignment
struct alignas(8) SumMax
{
    float maxval;
    float sum;
};

// forceinline helps avoid function call overhead
inline SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum =
        bigger_m.sum +
        smaller_m.sum * sycl::native::exp(smaller_m.maxval - bigger_m.maxval);
    return res;
}

void softmax_forward_online_kernel(float* out, const float* inp, int N, int C,
                                   const sycl::nd_item<1> &item) {

  sycl::sub_group warp = item.get_sub_group();

  int row = item.get_group(0) * warp.get_group_linear_range() +
            warp.get_group_linear_id();
  if (row >= N) {
    return;
  }

  // one row of inp, i.e. inp[row, :] of shape (C,)
  const float* x = inp + row * C;
  float* const y = out + row * C;

  // base case for the reduction
  SumMax sm_partial;
  sm_partial.maxval = -INFINITY;
  sm_partial.sum = 0.0f;

  // first, thread coarsening by directly accessing global memory in series
  for (int i = warp.get_local_linear_id(); i < C;
       i += warp.get_local_linear_range()) {
    sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
  }

  // second, the reduction (TODO)
  SumMax sm_total = sycl::ext::oneapi::experimental::reduce_over_group(warp,
                                            sm_partial, reduce_sum_max_op);
  SumMax sm_total = sycl::reduce_over_group(warp, sm_partial, reduce_sum_max_op);

  // divide the whole row by the sum
  for (int i = warp.get_local_linear_id(); i < C;
       i += warp.get_local_linear_range()) {
    y[i] = sycl::native::exp(x[i] - sm_total.maxval) / sm_total.sum;
  }
}
*/

void softmax_forward_online_kernel2(float* out, const float* inp, int N, int C,
                                    const sycl::nd_item<1> &item) {
  sycl::sub_group warp = item.get_sub_group();
  const int warpsPerBlock = item.get_local_range(0) /
                            warp.get_local_linear_range();
  int tid = item.get_local_id(0);

  if (tid >= C) {
    return;
  }

  int warpId = tid / warp.get_local_linear_range();
  int laneId = tid % warp.get_local_linear_range();
  int row = item.get_group(0) * warpsPerBlock + warpId;

  if (row >= N) {
    return;
  }

  const float* x = inp + row * C;
  float* const y = out + row * C;

  // merge calculating maxval and sumval in one loop
  // which is an arithmetic improvment from online softmax over normal softmax
  float maxval = -INFINITY, sumval = 0.0f, bigger;
  for (int i = laneId; i < C;
       i += warp.get_local_linear_range()) {
    // when updating the maxval, dynamically updates the previous sumval by
    // multiplying e^{previous_maxval - current_maxval}
    bigger = sycl::fmax(maxval, (float)(x[i]));
    sumval = sumval * sycl::native::exp(maxval - bigger) +
             sycl::native::exp(x[i] - bigger);
    maxval = bigger;
  }

  // use warp functions instead of cooperative groups for better readibility
  // calculate the warp wised maxval and sumval
  float offsetMaxval, offsetSumval;
  for (int offset = warp.get_local_linear_range() / 2;
       offset > 0; offset >>= 1) {
    sycl::group_barrier(warp);
    offsetMaxval = sycl::shift_group_left(warp, maxval, offset);
    offsetSumval = sycl::shift_group_left(warp, sumval, offset);
    if (offsetMaxval > maxval) {
      sumval *= sycl::native::exp(maxval - offsetMaxval);
      maxval = offsetMaxval;
    } else {
      offsetSumval *= sycl::native::exp(offsetMaxval - maxval);
    }
    sumval += offsetSumval;
  }

  // sync the warp wised maxval and sumval
  // which are also the maxval and sumval of one row in C
  maxval = sycl::select_from_group(warp, maxval, 0);
  sumval = sycl::select_from_group(warp, sumval, 0);

  for (int i = laneId; i < C;
       i += warp.get_local_linear_range()) {
    y[i] = sycl::native::exp(x[i] - maxval) / sumval;
  }
}

/*
void softmax_forward_online(sycl::queue &q, float* out, const float* inp, int N, int C,
                            int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  q.parallel_for(
      sycl::nd_range<1>(grid_size * block_size, block_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp_size)]] {
        softmax_forward_online_kernel(out, inp, N, C, item);
      }).wait();
}
*/

void softmax_forward_online2(sycl::queue &q, float* out, const float* inp, int N, int C,
                             int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  q.parallel_for(
      sycl::nd_range<1>(grid_size * block_size, block_size),
      [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(warp_size)]] {
        softmax_forward_online_kernel2(out, inp, N, C, item);
      }).wait();
}

// kernel version dispatch
void softmax_forward(int kernel_num, sycl::queue &q,
                     float* out, const float* inp, int N, int C,
                     const int block_size) {
  switch (kernel_num) {
    case 1:
      printf("kernel 1 not supported\n");
      //softmax_forward_online(q, out, inp, N, C, block_size);
      break;
    case 2:
      softmax_forward_online2(q, out, inp, N, C, block_size);
      break;
    default:
      printf("Invalid kernel number\n");
      exit(1);
  }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  // query the warp size
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int max_warp_size = *r;
  if (max_warp_size != warp_size) {
    printf("Please set the warp size in the main.cpp to %d\n.", max_warp_size);
    return 1;
  }

  srand(0);

  int B = 8;
  int T = 1024;
  int V = 50257;

  // create host memory of random numbers
  float* out = (float*)malloc(B * T * V * sizeof(float));
  float* inp = make_random_float(B * T * V);

  // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
  // and the tests are not actually meaningful.
  const int* outliers = make_random_int(B * T * 3, V);
  for(int k = 0; k < 3; ++k) {
    for(int j = 0; j < B * T; ++j) {
      inp[j * V + outliers[j*3 + k]] *= 20;
    }
  }

  // move to GPU
  float* d_out = sycl::malloc_device<float>(B * T * V, q);
  float* d_inp = sycl::malloc_device<float>(B * T * V, q);
  q.memcpy(d_inp, inp, B * T * V * sizeof(float)).wait();

  // read kernel_num from command line
  int kernel_num = 1;
  if (argc > 1) {
    kernel_num = atoi(argv[1]);
  }
  printf("Using kernel %d\n", kernel_num);

  
  softmax_forward_cpu(out, inp, B * T, V);
  {
    float max_el = -INFINITY;
    for(int i = 0; i <  B * T * V; ++i) {
      max_el = fmaxf(max_el, out[i]);
    }
    assert(max_el > 1e-4);
    printf("Largest output is: %f\n", max_el);
  }

  // first check the correctness of the kernel
  for (int j = warp_size; j <= 1024; j = j * 2) {
    int block_size = j;
    printf("Checking block size %d.\n", block_size);
    softmax_forward(kernel_num, q, d_out, d_inp, B * T, V, block_size);
    validate_result(d_out, out, "out", B * T * V, 1e-4f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  // time the kernel at different block sizes
  for (int j = warp_size; j <= 1024; j = j * 2) {
    int block_size = j;
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                          kernel_num, q, d_out, d_inp, B * T, V,
                                          block_size);
    printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
  }

  // free memory
  free(out);
  free(inp);
  free((void*)outliers);
  sycl::free(d_out, q);
  sycl::free(d_inp, q);

  return 0;
}
