// Complete LDA kernel with full algorithm
#include <omp.h>
#include <cmath>

#define EPS 1e-6f

// Helper functions for NaN checking
#pragma omp declare target
inline bool my_isnan(float x) {
  return x != x;
}
inline bool my_isinf(float x) {
  return x == __builtin_inff() || x == -__builtin_inff();
}
#pragma omp end declare target

// Digamma function
#pragma omp declare target
inline float Digamma(float x) {
  float result = 0.0f, xx, xx2, xx4;
  for ( ; x < 7.0f; ++x)
    result -= 1.0f / x;
  x -= 0.5f;
  xx = 1.0f / x;
  xx2 = xx * xx;
  xx4 = xx2 * xx2;
  result += logf(x) + 1.0f / 24.0f * xx2
    - 7.0f / 960.0f * xx4 + 31.0f / 8064.0f * xx4 * xx2
    - 127.0f / 30720.0f * xx4 * xx4;
  return result;
}

// Reduction function using shared memory
inline float ReduceSum(const float* vec, const int length, float* shared, int threadIdx_x, int blockDim_x) {
  // Each thread computes partial sum
  float val = 0.0f;
  for (int i = threadIdx_x; i < length; i += blockDim_x)
    val += vec[i];

  // Store in shared memory
  shared[threadIdx_x] = val;
  #pragma omp barrier

  // Reduce in shared memory
  for (int offset = blockDim_x / 2; offset > 0; offset /= 2) {
    if (threadIdx_x < offset) {
      shared[threadIdx_x] += shared[threadIdx_x + offset];
    }
    #pragma omp barrier
  }

  return shared[0];
}
#pragma omp end declare target

void EstepKernel(
  const int*__restrict__ cols,
  const int*__restrict__ indptr,
  const bool*__restrict__ vali,
  const float*__restrict__ counts,
  const bool init_gamma,
  const int num_cols,
  const int num_indptr,
  const int num_topics,
  const int num_iters,
  const float*__restrict__ alpha,
  const float*__restrict__ beta,
  float*__restrict__ gamma,
  float*__restrict__ grad_alpha,
  float*__restrict__ new_beta,
  float*__restrict__ train_losses,
  float*__restrict__ vali_losses,
  int*__restrict__ locks,
  const int block_cnt,
  const int block_dim)
{
  #pragma omp target teams num_teams(block_cnt) thread_limit(block_dim)
  {
    // Shared memory allocation per team
    float shared_mem[5000];  // Static allocation (num_topics * 4 + block_dim)
    float* _new_gamma = &shared_mem[0];
    float* _phi = &shared_mem[num_topics];
    float* _loss_vec = &shared_mem[num_topics * 2];
    float* _vali_phi_sum = &shared_mem[num_topics * 3];
    float* _reduce_shared = &shared_mem[num_topics * 4];

    #pragma omp parallel
    {
      int blockIdx_x = omp_get_team_num();
      int threadIdx_x = omp_get_thread_num();
      int blockDim_x = omp_get_num_threads();
      int gridDim_x = omp_get_num_teams();

      // Debug: print grid dimensions once per team
      if (threadIdx_x == 0 && blockIdx_x == 0) {
        printf("gridDim_x=%d, num_indptr=%d\n", gridDim_x, num_indptr);
      }

      float* _grad_alpha = grad_alpha + num_topics * blockIdx_x;

      for (int i = blockIdx_x; i < num_indptr; i += gridDim_x) {
        int beg = indptr[i], end = indptr[i + 1];
        float* _gamma = gamma + num_topics * i;

        // Debug: Print processing info for document range 200-220
        if (threadIdx_x == 0 && i >= 200 && i <= 220) {
          printf("Block %d processing doc %d: size=%d (beg=%d, end=%d)\n",
                 blockIdx_x, i, end-beg, beg, end);
        }

        // Initialize gamma
        if (init_gamma) {
          for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
            _gamma[j] = alpha[j] + (float)(end - beg) / num_topics;
          }
          #pragma omp barrier
          // Debug: check gamma after initialization
          if (threadIdx_x == 0 && i >= 200 && i <= 205) {
            float gamma_check = _gamma[0];
            printf("  Doc %d: after init, gamma[0] = %f, alpha[0]=%f, doc_size=%d\n",
                   i, gamma_check, alpha[0], end-beg);
          }
        }
        #pragma omp barrier

        // Initialize validation phi sum
        for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
          _vali_phi_sum[j] = 0.0f;

        // E-step iterations
        for (int iter = 0; iter < num_iters; ++iter) {
          // Initialize new gamma
          for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
            _new_gamma[k] = 0.0f;
          #pragma omp barrier

          // Process each word
          for (int k = beg; k < end; ++k) {
            const int w = cols[k];
            const bool _vali = vali[k];
            const float c = counts[k];

            // Compute phi
            if (!_vali || iter + 1 == num_iters) {
              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                float beta_val = beta[w * num_topics + l];
                float digamma_val = Digamma(_gamma[l]);
                float exp_val = expf(digamma_val);
                float phi_val = beta_val * exp_val;
                _phi[l] = phi_val;
                // Debug: check for inf immediately after assignment for specific indices
                if (l == 320 && i >= 200 && i <= 200 && iter == 0 && k == beg) {
                  printf("  Thread %d computes phi[320]: beta=%e, Digamma=%e, exp=%e, phi=%e\n",
                         threadIdx_x, beta_val, digamma_val, exp_val, phi_val);
                }
              }
              #pragma omp barrier

              // CRITICAL: Add another barrier before debug check to ensure all threads wait
              if (i >= 200 && i <= 200 && iter == 0 && k == beg) {
                #pragma omp barrier
              }

              // Debug: check phi and gamma for first word in iteration 0
              if (threadIdx_x == 0 && i >= 200 && i <= 200 && iter == 0 && k == beg) {
                float digamma_val = Digamma(_gamma[0]);
                float exp_val = expf(digamma_val);
                // Check for NaN/inf in phi array and find problematic gamma values
                int nan_count = 0, inf_count = 0;
                for (int jj = 0; jj < num_topics && inf_count < 5; jj++) {
                  if (my_isnan(_phi[jj])) nan_count++;
                  if (my_isinf(_phi[jj])) {
                    float beta_val = beta[w * num_topics + jj];
                    float exp_val = expf(Digamma(_gamma[jj]));
                    printf("  Doc %d: phi[%d]=inf, gamma[%d]=%e, Digamma=%e, exp=%e, beta[%d]=%e\n",
                           i, jj, jj, _gamma[jj], Digamma(_gamma[jj]), exp_val, w * num_topics + jj, beta_val);
                    inf_count++;
                  }
                }
                printf("  Doc %d iter %d word %d: gamma[0]=%f, phi NaN=%d, phi inf=%d\n",
                       i, iter, k, _gamma[0], nan_count, inf_count);
              }

              // Normalize phi using ReduceSum
              float phi_sum = ReduceSum(_phi, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
              float phi_sum_orig = phi_sum;  // Save original value
              phi_sum = fmaxf(phi_sum, EPS);  // Avoid division by zero

              // Debug: check phi_sum for first word
              if (threadIdx_x == 0 && i >= 200 && i <= 200 && iter == 0 && k == beg) {
                printf("  Doc %d iter %d word %d: phi_sum_orig=%e, phi_sum_clamped=%e\n",
                       i, iter, k, phi_sum_orig, phi_sum);
              }

              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                _phi[l] /= phi_sum;

                // Update gamma for train data and phi_sum for validation
                if (_vali)
                  _vali_phi_sum[l] += _phi[l] * c;
                else
                  _new_gamma[l] += _phi[l] * c;
              }
              #pragma omp barrier

              // Debug: check _new_gamma after first word
              if (threadIdx_x == 0 && i >= 200 && i <= 200 && iter == 0 && k == beg) {
                printf("  Doc %d iter %d after word %d: _new_gamma[0]=%e, _phi[0] after norm=%e\n",
                       i, iter, k, _new_gamma[0], _phi[0]);
              }
            }

            // Last iteration: update beta and compute loss
            if (iter + 1 == num_iters) {
              // Update beta for training data using atomics
              if (!_vali) {
                for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                  #pragma omp atomic
                  new_beta[w * num_topics + l] += _phi[l] * c;
                }
                #pragma omp barrier
              }

              // Compute loss vector
              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                float beta_val = fmaxf(beta[w * num_topics + l], EPS);
                float phi_val = fmaxf(_phi[l], EPS);
                // Clamp values to reasonable range to prevent overflow
                beta_val = fminf(beta_val, 1.0f);
                phi_val = fminf(phi_val, 1.0f);
                _loss_vec[l] = logf(beta_val) - logf(phi_val);
                _loss_vec[l] *= phi_val;
              }
              #pragma omp barrier

              float _loss = ReduceSum(_loss_vec, num_topics, _reduce_shared, threadIdx_x, blockDim_x) * c;
              if (threadIdx_x == 0) {
                // Check for NaN/inf before atomic add
                if (my_isnan(_loss) || my_isinf(_loss)) {
                  _loss = 0.0f;
                }
                // Debug: print loss values for doc range 200-220
                if (i >= 200 && i <= 220 && !_vali) {
                  printf("  Doc %d: word_loss = %f (word_idx=%d)\n", i, _loss, k);
                }
                if (_vali) {
                  #pragma omp atomic
                  vali_losses[i] += _loss;
                } else {
                  #pragma omp atomic
                  train_losses[i] += _loss;
                }
              }
              #pragma omp barrier
            }
            #pragma omp barrier
          }

          // Update gamma
          for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
            _gamma[k] = _new_gamma[k] + alpha[k];
          #pragma omp barrier
          // Debug: check gamma after first iteration
          if (threadIdx_x == 0 && i >= 200 && i <= 202 && iter == 0) {
            float new_gamma_sum = 0.0f;
            for (int jj = 0; jj < num_topics; jj++) new_gamma_sum += _new_gamma[jj];
            float gamma_sum_check = 0.0f;
            for (int jj = 0; jj < num_topics; jj++) gamma_sum_check += _gamma[jj];
            printf("  Doc %d iter %d: new_gamma_sum=%f, gamma_sum=%f, gamma[0]=%f\n",
                   i, iter, new_gamma_sum, gamma_sum_check, _gamma[0]);
          }
        }

        // Update gradient of alpha and loss from E[log(theta)]
        float gamma_sum = ReduceSum(_gamma, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
        gamma_sum = fmaxf(gamma_sum, EPS);  // Protect against zero/negative
        float digamma_sum = Digamma(gamma_sum);
        for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
          float gamma_val = fmaxf(_gamma[j], EPS);
          float Elogthetad = Digamma(gamma_val) - digamma_sum;
          _grad_alpha[j] += Elogthetad;
          _new_gamma[j] *= Elogthetad;
          _vali_phi_sum[j] *= Elogthetad;
        }

        // Compute final losses
        float train_loss = ReduceSum(_new_gamma, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
        float vali_loss = ReduceSum(_vali_phi_sum, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
        if (threadIdx_x == 0) {
          // Debug: print final loss for doc range 200-220
          if (i >= 200 && i <= 220) {
            printf("  Doc %d: final_train_loss = %f, gamma_sum=%f\n", i, train_loss, gamma_sum);
          }
          // Check for NaN/inf before atomic add
          if (!my_isnan(train_loss) && !my_isinf(train_loss)) {
            #pragma omp atomic
            train_losses[i] += train_loss;
          }
          if (!my_isnan(vali_loss) && !my_isinf(vali_loss)) {
            #pragma omp atomic
            vali_losses[i] += vali_loss;
          }
        }

        #pragma omp barrier
      }
    }
  }
}
