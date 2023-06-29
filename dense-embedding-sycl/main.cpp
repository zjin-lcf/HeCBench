#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>

template <typename T>
void reference(
    const T* input,
    const T* dense,
    T* output,
    int embedding_dim,
    int batch_size,
    const int* offset)
{
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] = 
          input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}

template <typename T>
void dense_esuhm(
    sycl::nd_item<1> &item,
    const T* input,
    const T* dense,
          T* output,
    int embedding_dim,
    const int* offset)
{
  const int batch_idx  = item.get_group(0); // each batch is handled by a block
  const int grain_size = item.get_local_range(0);
  const int tid = item.get_local_id(0);
  const int range = offset[batch_idx + 1] - offset[batch_idx];
  for (int idx = tid; idx < embedding_dim; idx += grain_size) {
    const T dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[offset[batch_idx] + nested_idx] =
        input[offset[batch_idx] + nested_idx] + dense_elem;
    }
  }
}

template <typename T>
void dense_esuhm2(
    sycl::nd_item<1> &item,
    const T* input,
    const T* dense,
          T* output,
    int embedding_dim,
    const int* offset)
{
  const int batch_idx = item.get_group(0);
  const int start = offset[batch_idx];
  const int range = offset[batch_idx + 1] - start;
  for (int idx = item.get_local_id(0); idx < embedding_dim;
           idx += item.get_local_range(0)) {
    const T dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[start + nested_idx] = input[start + nested_idx] + dense_elem;
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int batch_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  assert(nrows > batch_size * batch_size);

  printf("Number of rows in the embedding table: %d\n", nrows);
  printf("Batch size: %d\n", batch_size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  for (int ncols = 64; ncols <= 2048; ncols = ncols * 2) {

    printf("\nEmbedding dimension: %d\n", ncols);

    int input_size = nrows * ncols;  // same as output size
    size_t input_size_bytes = input_size * sizeof(float);

    int dense_size = batch_size * ncols ;
    int dense_size_bytes = dense_size * sizeof(float);

    int batch_size_bytes = (batch_size + 1) * sizeof(float);

    float *input, *dense, *output, *output_ref;
    input = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    dense = (float*) malloc (dense_size_bytes); // [batch_size x embedding_dim]
    output = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    output_ref = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    int *input_offset = (int*) malloc (batch_size_bytes);  // [batch_size]

    // meet a few conditions:
    // range > ncols
    // range + offset[i] < input_size
    // offset[i] > offset[i-1]
    srand(123);
    input_offset[0] = 0;
    for (int i = 1; i <= batch_size; i++)
      input_offset[i] = input_offset[i-1] + (rand() % batch_size + 1) * ncols;

    std::default_random_engine g (123);
    std::uniform_real_distribution<float> distr (-1.f, 1.f);
    for (int i = 0; i < dense_size; i++) {
      dense[i] = distr(g);
    }

    for (int i = 0; i < input_size; i++) {
      input[i] = distr(g);
      output_ref[i] = 0;
    }

    reference(input, dense, output_ref, ncols, batch_size, input_offset);

    float *d_input, *d_dense, *d_output;
    d_input = sycl::malloc_device<float>(input_size, q);
    q.memcpy(d_input, input, input_size_bytes);
    
    d_dense = sycl::malloc_device<float>(dense_size, q);
    q.memcpy(d_dense, dense, dense_size_bytes);

    d_output = sycl::malloc_device<float>(input_size, q);
    q.memset(d_output, 0, input_size_bytes);

    int* d_input_offset;
    d_input_offset = sycl::malloc_device<int>(batch_size+1, q);
    q.memcpy(d_input_offset, input_offset, batch_size_bytes);
    
    sycl::range<1> gws (batch_size * 256);
    sycl::range<1> lws (256);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class de1>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          dense_esuhm(item, d_input, d_dense, d_output, ncols, d_input_offset);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of dense embedding kernel (k1): %f (us)\n", (time * 1e-3f) / repeat);

    q.wait();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class de2>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          dense_esuhm2(item, d_input, d_dense, d_output, ncols, d_input_offset);
        });
      });
    }

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of dense embedding kernel (k2): %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(output, d_output, input_size_bytes).wait();

    bool ok = true;
    for (int i = 0; i < input_size; i++) {
      if (fabsf(output[i] - output_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    sycl::free(d_input, q);
    sycl::free(d_dense, q);
    sycl::free(d_output, q);
    sycl::free(d_input_offset, q);

    free(input);
    free(dense);
    free(output);
    free(output_ref);
    free(input_offset);
  }

  return 0;
}
