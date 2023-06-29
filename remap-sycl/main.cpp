#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define NUM_THREADS 256

void remap_kernel(dpct::device_pointer<const int> second_order,
                  dpct::device_pointer<const int> first_order,
                  int *output,
                  const int N, const int K,
                  sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i >= K) return;
  int idx = second_order[i];
  output[first_order[idx]] = i;
  for (idx++; idx < N && (i == K - 1 || idx != second_order[i + 1]); idx++) {
    output[first_order[idx]] = i;
  }
}

template <typename T>
void eval_remap(sycl::queue &q, const int N, const int repeat) {

  size_t input_size_bytes = N * sizeof(T);
  size_t output_size_bytes = N * sizeof(int);

  int *h_input = (int*) malloc (input_size_bytes);

#ifdef EXAMPLE
  h_input[0] = 1; h_input[1] = 3; h_input[2] = 5;
  h_input[3] = 1; h_input[4] = 5; h_input[5] = 7;
  h_input[6] = 9;
#else
  srand(123);
  for (int i = 0; i < N; i++) {
    h_input[i] = rand() % N;
  }
#endif

  int *h_output = (int*) malloc (output_size_bytes);

  long seq_time = 0,
       sort_time = 0,
       unique_time = 0,
       kernel_time = 0,
       copy_time = 0,
       alloc_time = 0,
       dealloc_time = 0;

  auto policy = oneapi::dpl::execution::make_device_policy(q);

  auto offload_start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    auto start = std::chrono::steady_clock::now();

    T *d_input = sycl::malloc_device<T>(N, q);

    int *d_output = sycl::malloc_device<int>(N, q);

    dpct::device_vector<int> order1(N), order2(N);

    auto end = std::chrono::steady_clock::now();
    alloc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();

    q.memcpy(d_input, h_input, input_size_bytes).wait();

    end = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();

    // Create two vectors of {0, 1, ..., N-1} on device
    dpct::iota(policy, order1.begin(), order1.end());
    dpct::iota(policy, order2.begin(), order2.end());

    end = std::chrono::steady_clock::now();
    seq_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Sort the input along with order vector. So now we know where each element
    // is permutated to. For example:
    //    input1 = 1,3,5,1,5,7,9
    //    order1 = 0,1,2,3,4,5,6
    // Now we have:
    //    output = 1,1,3,5,5,7,9
    //    order1 = 0,3,1,2,4,5,6
    start = std::chrono::steady_clock::now();

    auto buffer = dpct::get_device_pointer(d_input);
    dpct::sort(policy, buffer, buffer + N, order1.begin());

    //oneapi::dpl::sort_by_key(policy, d_input, d_input + N, order1.begin());

    end = std::chrono::steady_clock::now();
    sort_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Use consequent unique op to get another order_buffer
    //    input2 = 1,1,3,5,5,7,9
    //    order2 = 0,1,2,3,4,5,6
    // Now we have:
    //    output = 1,3,5,7,9
    //    order2 = 0,2,3,5,6
    start = std::chrono::steady_clock::now();

    auto result = dpct::unique(policy, buffer, buffer + N, order2.begin());

    end = std::chrono::steady_clock::now();
    unique_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    int K = result.first - buffer;
    if (i == 0) printf("Percentage of unique elements: %.1f %%\n", (float) K * 100 / N);

    // Compute the remapping. For example, for the number 1, if we look at
    // order2[0] and order2[1], we know that input2[0:2) are all 1. They are all
    // remapped to 0 in final input. And from order1, we know where they come from.
    sycl::range<1> gws ((K + NUM_THREADS - 1) / NUM_THREADS * NUM_THREADS);
    sycl::range<1> lws (NUM_THREADS);

    q.wait();
    start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &cgh) {
      auto order2_ptr = order2.data();
      auto order1_ptr = order1.data();
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        remap_kernel(order2_ptr, order1_ptr, d_output, N, K, item);
      });
    }).wait();

    end = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();

    q.memcpy(h_output, d_output, output_size_bytes).wait();

    end = std::chrono::steady_clock::now();
    copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();

    sycl::free(d_output, q);
    sycl::free(d_input, q);

    end = std::chrono::steady_clock::now();
    dealloc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();

  printf("Average offload time: %f (s)\n", offload_time * 1e-9f / repeat);
  printf("Average execution time of memory allocation : %f (us)\n", (alloc_time * 1e-3f) / repeat);
  printf("Average execution time of memory deallocation : %f (us)\n", (dealloc_time * 1e-3f) / repeat);
  printf("Average execution time of data copy : %f (us)\n", (copy_time * 1e-3f) / repeat);
  printf("Average execution time of oneDPL sequence : %f (us)\n", (seq_time * 1e-3f) / repeat);
  printf("Average execution time of oneDPL sort-by-key : %f (us)\n", (sort_time * 1e-3f) / repeat);
  printf("Average execution time of oneDPL unique-by-key : %f (us)\n", (unique_time * 1e-3f) / repeat);
  printf("Average execution time of remap kernel: %f (us)\n", (kernel_time * 1e-3f) / repeat);

  int cs1 = 0, cs2 = 0;
  for (int i = 0; i < N-1; i++) {
    cs1 ^= h_output[i] - h_output[i+1];
  }
  for (int i = 0; i < N; i++) {
    cs2 ^= h_output[i];
  }
  printf("Checksum: %d %d\n", cs1, cs2);

#ifdef EXAMPLE
  for (int i = 0; i < N; i++) {
    printf("%d ", h_output[i]);
  }
  printf("\n");
#endif

  free(h_output);
  free(h_input);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

#ifdef EXAMPLE
  const int N = 7;
#else
  const int N = atoi(argv[1]);
#endif
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // warmup and run
  for (int i = 0; i < 2; i++) {
    printf("\nRun %d\n", i);
    eval_remap<int>(q, N, repeat);
  }

  return 0;
}
