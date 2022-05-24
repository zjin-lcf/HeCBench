#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include "common.h"
#include "reference.h"

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int repeat) 
{
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // input
  buffer<float, 1> d_key (key, n*d);
  buffer<float, 1> d_value (value, n*d);
  buffer<float, 1> d_query (query, d);

  // intermediate
  buffer<float, 1> d_dot_product (n);
  buffer<float, 1> d_score (n);
  buffer<float, 1> d_exp_sum (1);

  // result
  float *output = (float*) malloc (d * sizeof(float));
  buffer<float, 1> d_output (d);

  range<1> n_gws ((n+255)/256*256);
  range<1> n_lws (256);
  range<1> d_gws ((d+255)/256*256);
  range<1> d_lws (256);

  for (int k = 0; k < repeat; k++) {
    q.submit([&] (handler &cgh) {
      auto exp_sum = d_exp_sum.get_access<sycl_discard_write>(cgh);
      cgh.fill(exp_sum, 0.f);
    });

    q.submit([&] (handler &cgh) {
      auto key = d_key.get_access<sycl_read>(cgh);
      auto query = d_query.get_access<sycl_read>(cgh);
      auto dot_product = d_dot_product.get_access<sycl_discard_write>(cgh);
      auto exp_sum = d_exp_sum.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class k1>(nd_range<1>(n_gws, n_lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          float sum = 0;
          for (int j = 0; j < d; j++)
            sum += key[i * d + j] * query[j];
          dot_product[i] = sum;
          auto atomic_obj_ref = ext::oneapi::atomic_ref<float, 
            ext::oneapi::memory_order::relaxed,
            ext::oneapi::memory_scope::device,
            access::address_space::global_space> (exp_sum[0]);
          atomic_obj_ref.fetch_add(sycl::exp(sum));
        }
      });
    });

    q.submit([&] (handler &cgh) {
      auto score = d_score.get_access<sycl_discard_write>(cgh);
      auto dot_product = d_dot_product.get_access<sycl_read>(cgh);
      auto exp_sum = d_exp_sum.get_access<sycl_read>(cgh);
      cgh.parallel_for<class k2>(nd_range<1>(n_gws, n_lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n)
          score[i] = sycl::exp(dot_product[i]) / exp_sum[0];
      });
    });

    q.submit([&] (handler &cgh) {
      auto output = d_output.get_access<sycl_discard_write>(cgh);
      auto score = d_score.get_access<sycl_read>(cgh);
      auto value = d_value.get_access<sycl_read>(cgh);
      cgh.parallel_for<class k3>(nd_range<1>(d_gws, d_lws), [=] (nd_item<1> item) {
        int j = item.get_global_id(0);
        if (j < d) {
          float sum = 0;
          for (int i = 0; i < n; i++)
            sum += score[i] * value[i * d + j];
          output[j] = sum;
        }
      });
    });
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_output.get_access<sycl_read>(cgh);
    cgh.copy(acc, output);
  }).wait();

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

  srand(2);
  for (int i = 0; i < n * d; i++) {
    key[i] = 0.1;
    value[i] = 0.3;
    if (rand() % 2)
      query[i % d] = value[i] + key[i] ;
    else
      query[i % d] = value[i] - key[i] ;
  }

  float* hout = attention_host(key, value, query, n, d);

  auto start = std::chrono::steady_clock::now();

  float* dout = attention_device(key, value, query, n, d, r);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offload time %f (s)\n", (time * 1e-9f));

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
