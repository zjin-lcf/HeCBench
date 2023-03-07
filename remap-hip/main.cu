#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/version.h>

#define NUM_THREADS 256

__global__
void remap_kernel(
  thrust::device_ptr<const int> second_order,
  thrust::device_ptr<const int> first_order,
  int *output,
  const int N,
  const int K)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= K) return;
  int idx = second_order[i];
  output[first_order[idx]] = i;
  for (idx++; idx < N && (i == K - 1 || idx != second_order[i + 1]); idx++) {
    output[first_order[idx]] = i;
  }
}

template <typename T>
void eval_remap(const int N, const int repeat) {

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

  T *d_input;
  hipMalloc((void**)&d_input, input_size_bytes);
  hipMemcpy(d_input, h_input, input_size_bytes, hipMemcpyHostToDevice);

  int *d_output;
  hipMalloc((void**)&d_output, output_size_bytes);

  auto start = std::chrono::steady_clock::now();

  // Create two vectors of {0, 1, ..., N-1} on device
  thrust::device_vector<int> order1(N), order2(N);
  thrust::sequence(order1.begin(), order1.end());
  thrust::sequence(order2.begin(), order2.end());

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of sequence : %f (us)\n", (time * 1e-3f));

  // Sort the input along with order vector. So now we know where each element
  // is permutated to. For example:
  //    input1 = 1,3,5,1,5,7,9
  //    order1 = 0,1,2,3,4,5,6
  // Now we have:
  //    output = 1,1,3,5,5,7,9
  //    order1 = 0,3,1,2,4,5,6
  start = std::chrono::steady_clock::now();

  auto buffer = thrust::device_pointer_cast(d_input);
  thrust::sort_by_key(buffer, buffer + N, order1.begin());

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of sort-by-key : %f (us)\n", (time * 1e-3f));

  // Use consequent unique op to get another order_buffer
  //    input2 = 1,1,3,5,5,7,9
  //    order2 = 0,1,2,3,4,5,6
  // Now we have:
  //    output = 1,3,5,7,9
  //    order2 = 0,2,3,5,6
  start = std::chrono::steady_clock::now();

  auto result = thrust::unique_by_key(buffer, buffer + N, order2.begin());

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of unique-by-key : %f (us)\n", (time * 1e-3f));

  int K = result.first - buffer;
  printf("Percentage of unique elements: %.1f %%\n", (float) K * 100 / N);

  // Compute the remapping. For example, for the number 1, if we look at
  // order2[0] and order2[1], we know that input2[0:2) are all 1. They are all
  // remapped to 0 in final input. And from order1, we know where they come from.
  dim3 grid ((K + NUM_THREADS - 1) / NUM_THREADS);
  dim3 block (NUM_THREADS);

  hipDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    remap_kernel<<<grid, block>>>(order2.data(), order1.data(), d_output, N, K);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of remap kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  hipMemcpy(h_output, d_output, output_size_bytes, hipMemcpyDeviceToHost);

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

  hipFree(d_output);
  hipFree(d_input);

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

  // warmup and run 
  for (int i = 0; i < 4; i++) {
    printf("\nRun %d\n", i);
    eval_remap<int>(N, repeat);
  }

  return 0;
}
