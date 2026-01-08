// Simplified LDA kernel WITHOUT debug code to test if debug is causing corruption
#include <omp.h>
#include <cmath>

#define EPS 1e-6f

// Helper functions for NaN checking
#pragma omp declare target
inline bool my_isnan(float x) {
  return x != x;
}
inline bool my_isinf(float x) {
  return x == __builtin_inff() || x == -__builtin_inff()
;
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
    float shared_mem[5000];
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

      float* _grad_alpha = grad_alpha + num_topics * blockIdx_x;

      for (int i = blockIdx_x; i < num_indptr; i += gridDim_x) {
        int beg = indptr[i], end = indptr[i + 1];
        float* _gamma = gamma + num_topics * i;

        // Initialize gamma
        if (init_gamma) {
          for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
            _gamma[j] = alpha[j] + (float)(end - beg) / num_topics;
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
              for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                _phi[l] = beta[w * num_topics + l] * expf(Digamma(_gamma[l]));
              #pragma omp barrier

              // Normalize phi using ReduceSum
              float phi_sum = ReduceSum(_phi, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
              phi_sum = fmaxf(phi_sum, EPS);

              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                _phi[l] /= phi_sum;

                if (_vali)
                  _vali_phi_sum[l] += _phi[l] * c;
                else
                  _new_gamma[l] += _phi[l] * c;
              }
              #pragma omp barrier
            }

            // Last iteration: update beta and compute loss
            if (iter + 1 == num_iters) {
              if (!_vali) {
                for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                  #pragma omp atomic
                  new_beta[w * num_topics + l] += _phi[l] * c;
                }
                #pragma omp barrier
              }

              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                float beta_val = fmaxf(beta[w * num_topics + l], EPS);
                float phi_val = fmaxf(_phi[l], EPS);
                beta_val = fminf(beta_val, 1.0f);
                phi_val = fminf(phi_val, 1.0f);
                _loss_vec[l] = logf(beta_val) - logf(phi_val);
                _loss_vec[l] *= phi_val;
              }
              #pragma omp barrier

              float _loss = ReduceSum(_loss_vec, num_topics, _reduce_shared, threadIdx_x, blockDim_x) * c;
              if (threadIdx_x == 0) {
                if (my_isnan(_loss) || my_isinf(_loss)) {
                  _loss = 0.0f;
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
        }

        // Update gradient of alpha and loss from E[log(theta)]
        float gamma_sum = ReduceSum(_gamma, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
        gamma_sum = fmaxf(gamma_sum, EPS);
        float digamma_sum = Digamma(gamma_sum);
        for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
          float gamma_val = fmaxf(_gamma[j], EPS);
          float Elogthetad = Digamma(gamma_val) - digamma_sum;
          _grad_alpha[j] += Elogthetad;
          _new_gamma[j] *= Elogthetad;
          _vali_phi_sum[j] *= Elogthetad;
        }

        float train_loss = ReduceSum(_new_gamma, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
        float vali_loss = ReduceSum(_vali_phi_sum, num_topics, _reduce_shared, threadIdx_x, blockDim_x);
        if (threadIdx_x == 0) {
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
