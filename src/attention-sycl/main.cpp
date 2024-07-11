#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int repeat) 
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // input
  float *d_key = sycl::malloc_device<float>(n*d, q);
  q.memcpy(d_key, key, sizeof(float) * n * d);

  float *d_value = sycl::malloc_device<float>(n*d, q);
  q.memcpy(d_value, value, sizeof(float) * n * d);

  float *d_query = sycl::malloc_device<float>(d, q);
  q.memcpy(d_query, query, sizeof(float) * d);

  // intermediate
  float *d_dot_product = sycl::malloc_device<float>(n, q);
  float *d_score = sycl::malloc_device<float>(n, q);
  float *d_exp_sum = sycl::malloc_device<float>(1, q);

  // result
  float *output = (float*) malloc (d * sizeof(float));
  float *d_output = sycl::malloc_device<float>(d, q);

  sycl::range<1> n_gws ((n+255)/256*256);
  sycl::range<1> n_lws (256);
  sycl::range<1> d_gws ((d+255)/256*256);
  sycl::range<1> d_lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int k = 0; k < repeat; k++) {
    q.memset(d_exp_sum, 0, sizeof(float));

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k1>(sycl::nd_range<1>(n_gws, n_lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          float sum = 0;
          for (int j = 0; j < d; j++)
            sum += d_key[i * d + j] * d_query[j];
          d_dot_product[i] = sum;
          auto ao = sycl::atomic_ref<float, 
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> (d_exp_sum[0]);
          ao.fetch_add(sycl::exp(sum));
        }
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k2>(sycl::nd_range<1>(n_gws, n_lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n)
          d_score[i] = sycl::exp(d_dot_product[i]) / d_exp_sum[0];
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k3>(sycl::nd_range<1>(d_gws, d_lws), [=] (sycl::nd_item<1> item) {
        int j = item.get_global_id(0);
        if (j < d) {
          float sum = 0;
          for (int i = 0; i < n; i++)
            sum += d_score[i] * d_value[i * d + j];
          d_output[j] = sum;
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);

  q.memcpy(output, d_output, sizeof(float) * d).wait();

  sycl::free(d_key, q);
  sycl::free(d_value, q);
  sycl::free(d_query, q);
  sycl::free(d_dot_product, q);
  sycl::free(d_score, q);
  sycl::free(d_exp_sum, q);
  sycl::free(d_output, q);

  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <rows> <columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int r = atoi(argv[3]);

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  for (int i = 0; i < n * d; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);

  float* dout = attention_device(key, value, query, n, d, r);

  float rmse = 0;
  for (int i = 0; i < d; i++) 
    rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
  printf("RMSE = %f\n", sqrtf(rmse / d));

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}
