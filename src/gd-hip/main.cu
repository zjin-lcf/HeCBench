#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <hip/hip_runtime.h>
#include "utils.h"

__global__ void 
L2_norm(const float *x, float* l2_norm, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(l2_norm, x[i]*x[i]);
}

__global__ void
compute (
    const float * __restrict__ x, 
          float * __restrict__ grad, 
    const int   * __restrict__ A_row_ptr,
    const int   * __restrict__ A_col_index,
    const float * __restrict__ A_value, 
    const int   * __restrict__ A_y_label,
    float * __restrict__ total_obj_val,
    int   * __restrict__ correct,
    int m ) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < m) {
    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
    for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective 
    float v = logf(1.f + expf(-xp * A_y_label[i]));
    atomicAdd(total_obj_val, v);

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) atomicAdd(correct, 1);

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      float temp = -accum * A_value[j] * A_y_label[i];
      atomicAdd(&grad[A_col_index[j]], temp);
    }
  }
}

__global__ void
update(float * __restrict__ x,
       const float * __restrict__ grad,
       int m, int n, float lambda, float alpha) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = grad[i] / (float)m + lambda * x[i]; 
    x[i] = x[i] - alpha * g;
  }
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

  float *d_x;
  hipMalloc((void**)&d_x, n * sizeof(float));
  hipMemcpy(d_x, x.data(), n * sizeof(float), hipMemcpyHostToDevice);

  float *d_grad; 
  hipMalloc((void**)&d_grad, n * sizeof(float));

  float *d_total_obj_val;
  hipMalloc((void**)&d_total_obj_val, sizeof(float));

  float *d_l2_norm;
  hipMalloc((void**)&d_l2_norm, sizeof(float));

  int *d_correct;
  hipMalloc((void**)&d_correct, sizeof(int));

  int *d_row_ptr;
  hipMalloc((void**)&d_row_ptr, A.row_ptr.size() * sizeof(int));
  hipMemcpy(d_row_ptr, A.row_ptr.data(), 
             A.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice);

  int *d_col_index;
  hipMalloc((void**)&d_col_index, A.col_index.size() * sizeof(int));
  hipMemcpy(d_col_index, A.col_index.data(), 
             A.col_index.size() * sizeof(int), hipMemcpyHostToDevice);

  float *d_value;
  hipMalloc((void**)&d_value, A.values.size() * sizeof(float));
  hipMemcpy(d_value, A.values.data(), 
             A.values.size() * sizeof(float), hipMemcpyHostToDevice);

  int *d_y_label;
  hipMalloc((void**)&d_y_label, A.y_label.size() * sizeof(int));
  hipMemcpy(d_y_label, A.y_label.data(), 
             A.y_label.size() * sizeof(int), hipMemcpyHostToDevice);

  dim3 grid ((m+255)/256);
  dim3 block (256);

  dim3 grid2 ((n+255)/256);
  dim3 block2 (256);

  float obj_val = 0.f;
  float train_error = 0.f;

  hipDeviceSynchronize();
  long long train_start = get_time();

  for (int k = 0; k < iters; k++) {

    // reset the training status
    float total_obj_val = 0.f;
    float l2_norm = 0.f;
    int correct = 0;

    hipMemcpy(d_total_obj_val, &total_obj_val, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_l2_norm, &l2_norm, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_correct, &correct, sizeof(int), hipMemcpyHostToDevice);

    //reset gradient vector
    std::fill(grad.begin(), grad.end(), 0.f);

    hipMemcpy(d_grad, grad.data(), n * sizeof(float), hipMemcpyHostToDevice);

    // compute the total objective, correct rate, and gradient
    compute<<<grid, block>>>(
        d_x, 
        d_grad, 
        d_row_ptr,
        d_col_index,
        d_value,
        d_y_label,
        d_total_obj_val,
        d_correct,
        m
      ); 

    // display training status for verification
    L2_norm<<<grid2, block2>>>(d_x, d_l2_norm, n);

    hipMemcpy(&total_obj_val, d_total_obj_val, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&l2_norm, d_l2_norm, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&correct, d_correct, sizeof(int), hipMemcpyDeviceToHost);

    obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
    train_error = 1.f-(correct/(float)m); 

    // update x (gradient does not need to be updated)
    update<<<grid2, block2>>>(d_x, d_grad, m, n, lambda, alpha);
  }

  hipDeviceSynchronize();
  long long train_end = get_time();
  printf("Training time takes %lf (s) for %d iterations\n\n",
         (train_end - train_start) * 1e-6, iters);

  // After 100 iterations, the expected obj_val and train_error are 0.3358405828 and 0.07433331013
  printf("object value = %f train_error = %f\n", obj_val, train_error);

  hipFree(d_row_ptr);
  hipFree(d_col_index);
  hipFree(d_value);
  hipFree(d_y_label);
  hipFree(d_x);
  hipFree(d_grad);
  hipFree(d_total_obj_val);
  hipFree(d_l2_norm);
  hipFree(d_correct);

  return 0; 
}
