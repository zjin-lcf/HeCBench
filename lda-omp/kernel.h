// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
//
#define EPS 1e-6f

#pragma omp declare target 
inline float ReduceSum(const float* vec, const int length) {
  float s = 0.f;
  // #pragma omp parallel for reduction (+:s)
  for (int i = 0; i < length; i++) 
    s += vec[i];
  return s;
}

// reference: http://web.science.mq.edu.au/~mjohnson/code/digamma.c
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

template<int numTeams, int numThreads, int smemSize>
void EstepKernel(
  const int*__restrict cols,
  const int*__restrict indptr, 
  const bool*__restrict vali,
  const float*__restrict counts,
  const bool init_gamma,
  const int num_cols,
  const int num_indptr, 
  const int num_topics,
  const int num_iters,
  const float*__restrict alpha,
  const float*__restrict beta,
  float*__restrict gamma,
  float*__restrict grad_alpha,
  float*__restrict new_beta, 
  float*__restrict train_losses,
  float*__restrict vali_losses,
  int*__restrict locks)
{  
  // storage for block
  #pragma omp target teams num_teams(numTeams) thread_limit(numThreads)
  {
    float shared_memory[smemSize];
    #pragma omp parallel 
    {
      const int blockIdx_x = omp_get_team_num();
      const int threadIdx_x = omp_get_thread_num();
      const int blockDim_x = numThreads;
      const int gridDim_x = numTeams;

      float*__restrict  _new_gamma = &shared_memory[0];
      float*__restrict  _phi = &shared_memory[num_topics];
      float*__restrict  _loss_vec = &shared_memory[num_topics * 2];
      float*__restrict  _vali_phi_sum = &shared_memory[num_topics * 3];

      float* _grad_alpha = grad_alpha + num_topics * blockIdx_x;

      for (int i = blockIdx_x; i < num_indptr; i += gridDim_x) {
        int beg = indptr[i], end = indptr[i + 1];
        float* _gamma = gamma + num_topics * i;
        if (init_gamma) {
          for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
            _gamma[j] = alpha[j] + (end - beg) / num_topics;
          }
        }
        #pragma omp barrier
        
        // initiate phi sum for validation data for computing vali loss 
        for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
          _vali_phi_sum[j] = 0.0f;

        // iterate E step
        for (int j = 0; j < num_iters; ++j) {
          // initialize new gamma
          for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
            _new_gamma[k] = 0.0f;
          #pragma omp barrier

          // compute phi from gamma
          for (int k = beg; k < end; ++k) {
            const int w = cols[k];  // word
            const bool _vali = vali[k];
            const float c = counts[k]; 
            // compute phi
            if (not _vali or j + 1 == num_iters) {
              for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                _phi[l] = beta[w * num_topics + l] * expf(Digamma(_gamma[l]));
              #pragma omp barrier
              
              // normalize phi and add it to new gamma and new beta
              float phi_sum = ReduceSum(_phi, num_topics);

              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                _phi[l] /= phi_sum;
                
                // update gamma for train data and phi_sum for computing loss
                if (_vali) 
                  _vali_phi_sum[l] += _phi[l] * c;
                else
                  _new_gamma[l] += _phi[l] * c;
              
              }
              #pragma omp barrier
            }
            
            if (j + 1 == num_iters) {
              // update beta for train data
              if (not _vali) {
                // write access of w th vector of new_beta 
                if (threadIdx_x == 0) {
                  // while (atomicCAS(&locks[w], 0, 1)) {}
                  int v;
                  do { 
                    #pragma omp atomic capture 
                    { v = locks[w]; locks[w] = v == 0 ? 1 : v; }
                  } while (v);
                } 

                #pragma omp barrier
                for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                  new_beta[w * num_topics + l] += _phi[l] * c;
                #pragma omp barrier

                // release lock
                if (threadIdx_x == 0) locks[w] = 0;
                #pragma omp barrier
              }
              
              // comput loss and reset shared mem
              // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
              for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                _loss_vec[l] = logf(fmaxf(beta[w * num_topics + l], EPS));
                _loss_vec[l] -= logf(fmaxf(_phi[l], EPS));
                _loss_vec[l] *= _phi[l];
              }
              #pragma omp barrier
              float _loss = ReduceSum(_loss_vec, num_topics) * c;
              if (threadIdx_x == 0) {
                if (_vali) 
                  vali_losses[blockIdx_x] += _loss;
                else
                  train_losses[blockIdx_x] += _loss;
              }
              #pragma omp barrier

            }
            #pragma omp barrier
          }

          // update gamma
          for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
            _gamma[k] = _new_gamma[k] + alpha[k];
          #pragma omp barrier
        }

        // update gradient of alpha and loss from E[log(theta)]
        float gamma_sum = ReduceSum(_gamma, num_topics);
        for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
          float Elogthetad = Digamma(_gamma[j]) - Digamma(gamma_sum);
          _grad_alpha[j] += Elogthetad;
          _new_gamma[j] *= Elogthetad;
          _vali_phi_sum[j] *= Elogthetad;
        }
        
        // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
        float train_loss = ReduceSum(_new_gamma, num_topics);
        float vali_loss = ReduceSum(_vali_phi_sum, num_topics);
        if (threadIdx_x == 0) {
          train_losses[blockIdx_x] += train_loss;
          vali_losses[blockIdx_x] += vali_loss;
        }

        #pragma omp barrier
      } 
    } 
  } 
}
