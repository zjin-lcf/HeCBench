#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "atomics.h"
#include "reference.h"

#define GPU_NUM_THREADS 256

template <typename T>
void BlockReduce(T &input1, sycl::nd_item<3> &item) {
  input1 = sycl::reduce_over_group(item.get_group(), input1, sycl::plus<T>());
}

// begin of accuracy_kernel
void accuracy_kernel(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const int N,
    const int D,
    const int top_k,
    const float* __restrict Xdata,
    const int* __restrict labelData,
    int* accuracy)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int count = 0;
      for (int row = item.get_group(2); row < N; row += item.get_group_range(2)) {
         const int label = labelData[row];
         const float label_pred = Xdata[row * D + label];
         int ngt = 0;
         for (int col = item.get_local_id(2); col < D; col += item.get_local_range(2)) {
           const float pred = Xdata[row * D + col];
           if (pred > label_pred || (pred == label_pred && col <= label)) {
             ++ngt;
           }
         }
         BlockReduce(ngt, item);
         if (ngt <= top_k) {
           ++count;
         }
         item.barrier(sycl::access::fence_space::local_space);
       }
       if (item.get_local_id(2) == 0) {
         atomicAdd(accuracy, count);
       }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
// end of accuracy_kernel
 
int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int data_size = nrows * ndims;

  const int label_size_bytes = nrows * sizeof(int); 
  const size_t data_size_bytes = data_size * sizeof(float);

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims; 

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int count_ref = reference(nrows, ndims, top_k, data, label);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_label = sycl::malloc_device<int>(nrows, q);
  q.memcpy(d_label, label, label_size_bytes);

  float *d_data = sycl::malloc_device<float>(data_size, q);
  q.memcpy(d_data, data, data_size_bytes);

  int *d_count = sycl::malloc_device<int>(1, q);

  q.wait();
  sycl::range<3> lws (1, 1, GPU_NUM_THREADS);

  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {

    printf("Grid size is %d\n", ngrid);
    sycl::range<3> gws (1, 1, ngrid * GPU_NUM_THREADS);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_count, 0, sizeof(int));
      accuracy_kernel(q, gws, lws, 0, nrows, ndims, top_k, d_data, d_label, d_count);
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

    int count;
    q.memcpy(&count, d_count, sizeof(int)).wait();
    bool ok = (count == count_ref);
    printf("%s\n", ok ? "PASS" : "FAIL");
    // printf("Accuracy = %f\n", (float)count / nrows);
  }

  sycl::free(d_label, q);
  sycl::free(d_data, q);
  sycl::free(d_count, q);

  free(label);
  free(data);

  return 0;
}
