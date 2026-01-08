// Simplified LDA kernel to test if we hit LLVM backend bug
#include <omp.h>
#include <cmath>

#define EPS 1e-6f

// Simple digamma function
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
  // Allocate shared memory per team (simplified - not using dynamic extern shared)
  const int shared_size = num_topics * 4 + block_dim;  // for reduction

  #pragma omp target teams num_teams(block_cnt) thread_limit(block_dim)
  {
    float shared_mem[4000];  // Static allocation (assumes num_topics * 4 < 4000)
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

        if (init_gamma) {
          for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
            _gamma[j] = alpha[j] + (float)(end - beg) / num_topics;
          }
        }
        #pragma omp barrier

        // Initialize validation phi sum
        for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
          _vali_phi_sum[j] = 0.0f;

        // Simplified E-step iteration (just structure, not full algorithm)
        for (int iter = 0; iter < num_iters; ++iter) {
          // Initialize new gamma
          for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
            _new_gamma[k] = 0.0f;
          #pragma omp barrier

          // Compute phi from gamma (simplified)
          for (int k = beg; k < end; ++k) {
            const int w = cols[k];
            const bool _vali = vali[k];
            const float c = counts[k];

            if (!_vali || iter + 1 == num_iters) {
              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                _phi[l] = beta[w * num_topics + l] * expf(Digamma(_gamma[l]));
              }
              #pragma omp barrier

              // Simplified reduction and update
              float phi_sum = 0.0f;
              for (int l = 0; l < num_topics; l++)
                phi_sum += _phi[l];

              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                _phi[l] /= (phi_sum + EPS);
                if (_vali)
                  _vali_phi_sum[l] += _phi[l] * c;
                else
                  _new_gamma[l] += _phi[l] * c;
              }
              #pragma omp barrier

              // Update beta with atomic lock (simplified)
              if (iter + 1 == num_iters && !_vali) {
                if (threadIdx_x == 0) {
                  int expected = 0;
                  #pragma omp atomic capture
                  { expected = locks[w]; locks[w] = 1; }
                  while (expected != 0) {
                    #pragma omp atomic read
                    expected = locks[w];
                  }
                }
                #pragma omp barrier

                for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                  new_beta[w * num_topics + l] += _phi[l] * c;
                #pragma omp barrier

                if (threadIdx_x == 0) {
                  #pragma omp atomic write
                  locks[w] = 0;
                }
                #pragma omp barrier
              }
            }
            #pragma omp barrier
          }

          // Update gamma
          for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
            _gamma[k] = _new_gamma[k] + alpha[k];
          #pragma omp barrier
        }
      }
    }
  }
}
