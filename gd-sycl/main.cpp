#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "common.h"
#include "utils.h"

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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  buffer<float, 1> d_x(x.data(), n);

  buffer<float, 1> d_grad(n);

  buffer<float, 1> d_total_obj_val(1);

  buffer<float, 1> d_l2_norm(1);

  buffer<int, 1> d_correct(1);

  buffer<int, 1> d_row_ptr(A.row_ptr.data(), A.row_ptr.size());

  buffer<int, 1> d_col_index(A.col_index.data(), A.col_index.size());

  buffer<float, 1> d_value(A.values.data(), A.values.size());

  buffer<int, 1> d_y_label(A.y_label.data(), A.y_label.size());

  range<1> gws((m+255)/256*256);
  range<1> lws (256);

  range<1> gws2((n+255)/256*256);
  range<1> lws2 (256);

  float obj_val = 0.f;
  float train_error = 0.f;

  q.wait();
  long long train_start = get_time();

  for (int k = 0; k < iters; k++) {

    // reset the training status
    float total_obj_val = 0.f;
    float l2_norm = 0.f;
    int correct = 0;

    q.submit([&] (handler &cgh) {
      auto acc = d_total_obj_val.get_access<sycl_discard_write>(cgh);
      cgh.copy(&total_obj_val, acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_l2_norm.get_access<sycl_discard_write>(cgh);
      cgh.copy(&l2_norm, acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_correct.get_access<sycl_discard_write>(cgh);
      cgh.copy(&correct, acc);
    });

    //reset gradient vector
    std::fill(grad.begin(), grad.end(), 0.f);

    q.submit([&] (handler &cgh) {
      auto acc = d_grad.get_access<sycl_discard_write>(cgh);
      cgh.copy(grad.data(), acc);
    });
    
    // compute the total objective, correct rate, and gradient
    q.submit([&] (handler &cgh) {
      auto x = d_x.get_access<sycl_read>(cgh);
      auto grad = d_grad.get_access<sycl_read_write>(cgh);
      auto A_row_ptr = d_row_ptr.get_access<sycl_read>(cgh);
      auto A_col_index = d_col_index.get_access<sycl_read>(cgh);
      auto A_value = d_value.get_access<sycl_read>(cgh);
      auto A_y_label = d_y_label.get_access<sycl_read>(cgh);
      auto total_obj_val = d_total_obj_val.get_access<sycl_read_write>(cgh);
      auto correct = d_correct.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class compute>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < m) {
          // Simple sparse matrix multiply x' = A * x
          float xp = 0.f;
          for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
            xp += A_value[j] * x[A_col_index[j]];
          }

          // compute objective 
          float v = sycl::log(1+sycl::exp(-1*A_y_label[i]*xp));
          auto atomic_obj_ref = atomic_ref<float,
            memory_order::relaxed, memory_scope::device,
            access::address_space::global_space> (total_obj_val[0]);
          atomic_obj_ref.fetch_add(v);

          // compute errors
          float prediction = 1.f/(1.f + sycl::exp(-xp));
          int t = (prediction >= 0.5f) ? 1 : -1;
          if (A_y_label[i] == t) {
            auto atomic_correct_ref = atomic_ref<int,
              memory_order::relaxed, memory_scope::device,
              access::address_space::global_space> (correct[0]);
            atomic_correct_ref.fetch_add(1);
	  }

          // compute gradient at x
          float accum = sycl::exp(-A_y_label[i] * xp);
          accum = accum / (1.f + accum);
          for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
            float temp = -accum*A_value[j]*A_y_label[i];
            auto atomic_grad_ref = atomic_ref<float,
              memory_order::relaxed, memory_scope::device,
              access::address_space::global_space> (grad[A_col_index[j]]);
            atomic_grad_ref.fetch_add(temp);
          }
        }
      }); 
    }); 

    // display training status for verification
    q.submit([&] (handler &cgh) {
      auto x = d_x.get_access<sycl_read>(cgh);
      auto l2_norm = d_l2_norm.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class norm>(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          auto atomic_l2norm_ref = atomic_ref<float,
            memory_order::relaxed, memory_scope::device,
            access::address_space::global_space> (l2_norm[0]);
          atomic_l2norm_ref.fetch_add(x[i]*x[i]);
        }
      });
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_total_obj_val.get_access<sycl_read>(cgh);
      cgh.copy(acc, &total_obj_val);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_l2_norm.get_access<sycl_read>(cgh);
      cgh.copy(acc, &l2_norm);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_correct.get_access<sycl_read>(cgh);
      cgh.copy(acc, &correct);
    });
    
    q.wait();

    obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
    train_error = 1.f-(correct/(float)m); 

    // update x (gradient does not need to be updated)
    q.submit([&] (handler &cgh) {
      auto x = d_x.get_access<sycl_read_write>(cgh);
      auto grad = d_grad.get_access<sycl_read>(cgh);
      cgh.parallel_for<class update>(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          float g = grad[i] / (float)m + lambda * x[i]; 
          x[i] = x[i] - alpha * g;
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

  return 0; 
}
