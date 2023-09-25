#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "utils.h"

template <typename T>
inline T atomicAdd(T *val, T operand)
{
  auto atm = sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  return atm.fetch_add(operand);
}

int main(int argc, const char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <path to file> <lambda> <alpha> <repeat>\n", argv[0]);
    return 1;
  }
  const std::string file_path = argv[1]; 
  const float lambda = atof(argv[2]);
  const float alpha = atof(argv[3]);
  const int iters = atof(argv[4]);

  //store the problem data in variable A and the data is going to be normalized
  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m; // observations
  const int n = A.n; // features

  std::vector<float> x(n, 0.f);
  std::vector<float> grad (n);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_x = sycl::malloc_device<float>(n, q);
  q.memcpy(d_x, x.data(), n * sizeof(float));

  float *d_grad = sycl::malloc_device<float>(n, q);

  float *d_total_obj_val = sycl::malloc_device<float>(1, q);

  float *d_l2_norm = sycl::malloc_device<float>(1, q);

  int *d_correct = sycl::malloc_device<int>(1, q);

  int *d_row_ptr = sycl::malloc_device<int>(A.row_ptr.size(), q);
  q.memcpy(d_row_ptr, A.row_ptr.data(), A.row_ptr.size() * sizeof(int));

  int *d_col_index = sycl::malloc_device<int>(A.col_index.size(), q);
  q.memcpy(d_col_index, A.col_index.data(), A.col_index.size() * sizeof(int));

  float *d_value = sycl::malloc_device<float>(A.values.size(), q);
  q.memcpy(d_value, A.values.data(), A.values.size() * sizeof(float));

  int *d_y_label = sycl::malloc_device<int>(A.y_label.size(), q);
  q.memcpy(d_y_label, A.y_label.data(), A.y_label.size() * sizeof(int));

  sycl::range<1> gws((m+255)/256*256);
  sycl::range<1> lws (256);

  sycl::range<1> gws2((n+255)/256*256);
  sycl::range<1> lws2 (256);

  float obj_val = 0.f;
  float train_error = 0.f;

  q.wait();
  long long train_start = get_time();

  for (int k = 0; k < iters; k++) {

    // reset the training status
    float total_obj_val = 0.f;
    float l2_norm = 0.f;
    int correct = 0;

    q.memcpy(d_total_obj_val, &total_obj_val, sizeof(float));
    q.memcpy(d_l2_norm, &l2_norm, sizeof(float));
    q.memcpy(d_correct, &correct, sizeof(int));

    //reset gradient vector
    std::fill(grad.begin(), grad.end(), 0.f);

    q.memcpy(d_grad, grad.data(), n * sizeof(float));
    
    // compute the total objective, correct rate, and gradient
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class compute>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < m) {
          // Simple sparse matrix multiply x' = A * x
          float xp = 0.f;
          for( int j = d_row_ptr[i]; j < d_row_ptr[i+1]; ++j){
            xp += d_value[j] * d_x[d_col_index[j]];
          }

          // compute objective 
          float v = sycl::log(1.f + sycl::exp(-xp * d_y_label[i]));
          atomicAdd(d_total_obj_val, v);

          // compute errors
          float prediction = 1.f / (1.f + expf(-xp));
          int t = (prediction >= 0.5f) ? 1 : -1;
          if (d_y_label[i] == t) {
            atomicAdd(d_correct, 1);
	  }

          // compute gradient at x
          float accum = sycl::exp(-d_y_label[i] * xp);
          accum = accum / (1.f + accum);
          for(int j = d_row_ptr[i]; j < d_row_ptr[i+1]; ++j){
            float temp = -accum * d_value[j] * d_y_label[i];
            atomicAdd(d_grad+d_col_index[j], temp);
          }
        }
      }); 
    }); 

    // display training status for verification
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class norm>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          atomicAdd(d_l2_norm, d_x[i]*d_x[i]);
        }
      });
    });

    q.memcpy(&total_obj_val, d_total_obj_val, sizeof(float));
    q.memcpy(&l2_norm, d_l2_norm, sizeof(float));
    q.memcpy(&correct, d_correct, sizeof(int));
    
    q.wait();

    obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
    train_error = 1.f-(correct/(float)m); 

    // update x (gradient does not need to be updated)
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class update>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          float g = d_grad[i] / (float)m + lambda * d_x[i];
          d_x[i] = d_x[i] - alpha * g;
        }
      });
    });
  }

  q.wait();
  long long train_end = get_time();
  printf("Training time takes %lf (s) for %d iterations\n\n", 
         (train_end - train_start) * 1e-6, iters);

  // After 100 iterations, the expected obj_val and train_error are 0.3358405828 and 0.07433331013
  printf("object value = %f train_error = %f\n", obj_val, train_error);

  sycl::free(d_row_ptr, q);
  sycl::free(d_col_index, q);
  sycl::free(d_value, q);
  sycl::free(d_y_label, q);
  sycl::free(d_x, q);
  sycl::free(d_grad, q);
  sycl::free(d_total_obj_val, q);
  sycl::free(d_l2_norm, q);
  sycl::free(d_correct, q);

  return 0; 
}
