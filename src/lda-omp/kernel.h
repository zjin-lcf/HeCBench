// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
//
#include <omp.h>
#define WARP_SIZE 32
#define EPS 1e-6f

// OpenMP version - use reduction instead of warp shuffles
#pragma omp declare target
inline float ReduceSum(const float* vec, const int length, float* shared, int threadIdx_x, int blockDim_x) {

  int warp =  threadIdx_x / WARP_SIZE;
  int lane = threadIdx_x % WARP_SIZE;

  // partial sum
  float val = 0.0f;
  for (int i = threadIdx_x; i < length; i += blockDim_x)
    val += vec[i];

  // Use shared memory for reduction instead of warp shuffles
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
  // storage for block
  extern __shared__ float shared_memory[];
  float*__restrict__  _new_gamma = &shared_memory[0];
  float*__restrict__  _phi = &shared_memory[num_topics];
  float*__restrict__  _loss_vec = &shared_memory[num_topics * 2];
  float*__restrict__  _vali_phi_sum = &shared_memory[num_topics * 3];

  float* _grad_alpha = grad_alpha + num_topics * blockIdx.x;

  for (int i = blockIdx.x; i < num_indptr; i += gridDim.x) {
    int beg = indptr[i], end = indptr[i + 1];
    float* _gamma = gamma + num_topics * i;
    if (init_gamma) {
      for (int j = threadIdx.x; j < num_topics; j += blockDim.x) {
        _gamma[j] = alpha[j] + (end - beg) / num_topics;
      }
    }
    __syncthreads();
    
    // initiate phi sum for validation data for computing vali loss 
    for (int j = threadIdx.x; j < num_topics; j += blockDim.x)
      _vali_phi_sum[j] = 0.0f;

    // iterate E step
    for (int j = 0; j < num_iters; ++j) {
      // initialize new gamma
      for (int k = threadIdx.x; k < num_topics; k += blockDim.x)
        _new_gamma[k] = 0.0f;
      __syncthreads();

      // compute phi from gamma
      for (int k = beg; k < end; ++k) {
        const int w = cols[k];  // word
        const bool _vali = vali[k];
        const float c = counts[k]; 
        // compute phi
        if (not _vali or j + 1 == num_iters) {
          for (int l = threadIdx.x; l < num_topics; l += blockDim.x)
            _phi[l] = beta[w * num_topics + l] * expf(Digamma(_gamma[l]));
          __syncthreads();
          
          // normalize phi and add it to new gamma and new beta
          float phi_sum = ReduceSum(_phi, num_topics);

          for (int l = threadIdx.x; l < num_topics; l += blockDim.x) {
            _phi[l] /= phi_sum;
            
            // update gamma for train data and phi_sum for computing loss
            if (_vali) 
              _vali_phi_sum[l] += _phi[l] * c;
            else
              _new_gamma[l] += _phi[l] * c;
          
          }
          __syncthreads();
        }
        
        if (j + 1 == num_iters) {
          // update beta for train data
          if (not _vali) {
            // write access of w th vector of new_beta 
            if (threadIdx.x == 0) {
              while (atomicCAS(&locks[w], 0, 1)) {}
            } 

            __syncthreads();
            for (int l = threadIdx.x; l < num_topics; l += blockDim.x)
              new_beta[w * num_topics + l] += _phi[l] * c;
            __syncthreads();

            // release lock
            if (threadIdx.x == 0) locks[w] = 0;
            __syncthreads();
          }
          
          // comput loss and reset shared mem
          // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
          for (int l = threadIdx.x; l < num_topics; l += blockDim.x) {
            _loss_vec[l] = logf(fmaxf(beta[w * num_topics + l], EPS));
            _loss_vec[l] -= logf(fmaxf(_phi[l], EPS));
            _loss_vec[l] *= _phi[l];
          }
          __syncthreads();
          float _loss = ReduceSum(_loss_vec, num_topics) * c;
          if (threadIdx.x == 0) {
            if (_vali) 
              vali_losses[blockIdx.x] += _loss;
            else
              train_losses[blockIdx.x] += _loss;
          }
          __syncthreads();

        }
        __syncthreads();
      }

      // update gamma
      for (int k = threadIdx.x; k < num_topics; k += blockDim.x)
        _gamma[k] = _new_gamma[k] + alpha[k];
      __syncthreads();
    }

    // update gradient of alpha and loss from E[log(theta)]
    float gamma_sum = ReduceSum(_gamma, num_topics);
    for (int j = threadIdx.x; j < num_topics; j += blockDim.x) {
      float Elogthetad = Digamma(_gamma[j]) - Digamma(gamma_sum);
      _grad_alpha[j] += Elogthetad;
      _new_gamma[j] *= Elogthetad;
      _vali_phi_sum[j] *= Elogthetad;
    }
    
    // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
    float train_loss = ReduceSum(_new_gamma, num_topics);
    float vali_loss = ReduceSum(_vali_phi_sum, num_topics);
    if (threadIdx.x == 0) {
      train_losses[blockIdx.x] += train_loss;
      vali_losses[blockIdx.x] += vali_loss;
    }

    __syncthreads();
  } 
}
