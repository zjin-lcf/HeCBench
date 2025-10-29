#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "utils.h"
#include "reference.h"

void L2_norm(const int numTeams, const int numThreads,
             const float *x, float* l2_norm, int n)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < n; ++i) {
    #pragma omp atomic update
    l2_norm[0] += x[i] * x[i];
  }
}

// begin of compute
void compute (
    const int numTeams,
    const int numThreads,
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
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads) 
  for (int i = 0; i < m; ++i) {

    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
         
    for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j) {
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective 
    float v = logf(1.f + expf(-xp * A_y_label[i]));
    #pragma omp atomic update
    total_obj_val[0] += v;

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) {
      #pragma omp atomic update
      correct[0]++;
    }

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j) {
      float temp = -accum * A_value[j] * A_y_label[i];
      #pragma omp atomic update
      grad[A_col_index[j]] += temp;
    }
  }
}
// end of compute

void update(
    const int numTeams,
    const int numThreads,
    float * __restrict__ x,
    const float * __restrict__ grad,
    int m, int n, float lambda, float alpha) 
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads) 
  for (int i = 0; i < n; ++i) {
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

  float *d_grad        = grad.data();
  float *d_x           = x.data();
  int   *d_row_ptr   = A.row_ptr.data(); 
  int   *d_col_index = A.col_index.data();
  float *d_value     = A.values.data();
  int   *d_y_label   = A.y_label.data();

  float total_obj_val[1];
  float l2_norm[1];
  int   correct[1];

  const int numTeams = ((m+255)/256);
  const int numThreads = 256;

  const int numTeams2 =  ((n+255)/256);
  const int numThreads2 = 256;

  #pragma omp target data map (to: d_x[0:n], \
                                   d_row_ptr[0:A.row_ptr.size()], \
                                   d_value[0:A.values.size()], \
                                   d_col_index[0:A.col_index.size()], \
                                   d_y_label[0:A.y_label.size()]) \
                          map (alloc: d_grad[0:n], \
                                      total_obj_val[0:1], \
                                      l2_norm[0:1], \
                                      correct[0:1])
  {
    long long train_start = get_time();

    float obj_val = 0.f;
    float train_error = 0.f;

    for (int k = 0; k < iters; k++) {

      // reset the training status
      total_obj_val[0] = 0.f;
      l2_norm[0] = 0.f;
      correct[0] = 0;
      
      #pragma omp target update to (total_obj_val[0:1])
      #pragma omp target update to (l2_norm[0:1])
      #pragma omp target update to (correct[0:1])

      //reset gradient vector
      std::fill(grad.begin(), grad.end(), 0.f);

      #pragma omp target update to (d_grad[0:n])
      
      // compute the total objective, correct rate, and gradient
      compute(
          numTeams,
          numThreads,
          d_x, 
          d_grad, 
          d_row_ptr,
          d_col_index,
          d_value,
          d_y_label,
          total_obj_val,
          correct,
          m
        ); 

      // display training status for verification
      L2_norm(numTeams2, numThreads2, d_x, l2_norm, n);

      #pragma omp target update from (total_obj_val[0:1])
      #pragma omp target update from (l2_norm[0:1])
      #pragma omp target update from (correct[0:1])

      obj_val = total_obj_val[0] / (float)m + 0.5f * lambda * l2_norm[0];
      train_error = 1.f - (correct[0]/(float)m); 

      // update x (gradient does not need to be updated)
      update(numTeams2, numThreads2, d_x, d_grad, m, n, lambda, alpha);
    }
    
    long long train_end = get_time();
    printf("Training time takes %lf (s) for %d iterations\n\n",
            (train_end - train_start) * 1e-6, iters);
    
    printf("object value = %f train_error = %f\n", obj_val, train_error);

    reference(A, x, grad, m, n, iters, alpha, lambda, obj_val, train_error);
  }

  return 0; 
}
