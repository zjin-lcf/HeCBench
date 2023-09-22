// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
//
#define WARP_SIZE 32
#define EPS 1e-6f

inline int atomicCAS(int &val, int expected, int desired)
{
  int expected_value = expected;
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

inline
float ReduceSum(sycl::nd_item<1> &item,
                float*__restrict shared,
                const float*__restrict vec,
                const int length)
{
  // figure out the warp/ position inside the warp
  int threadIdx_x = item.get_local_id(0);
  int blockDim_x = item.get_local_range(0);
  auto sg = item.get_sub_group();

  int warp =  threadIdx_x / WARP_SIZE;
  int lane = threadIdx_x % WARP_SIZE;

  // paritial sum
  float val = 0.0f;
  for (int i = threadIdx_x; i < length; i += blockDim_x)
    val += vec[i];

  //val = warp_reduce_sum(val);
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += sg.shuffle_down(val, offset);
  }

  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = val;
  }
  item.barrier(sycl::access::fence_space::local_space);

  // if we we don't have multiple warps, we're done
  if (blockDim_x <= WARP_SIZE) {
    return shared[0];
  }

  // otherwise reduce again in the first warp
  val = (threadIdx_x < blockDim_x / WARP_SIZE) ? shared[lane]: 0.0f;
  if (warp == 0) {
    // val = warp_reduce_sum(val);
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val += sg.shuffle_down(val, offset);
    }
    // broadcast back to shared memory
    if (threadIdx_x == 0) {
        shared[0] = val;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
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
  result += sycl::log(x) + 1.0f / 24.0f * xx2
    - 7.0f / 960.0f * xx4 + 31.0f / 8064.0f * xx4 * xx2
    - 127.0f / 30720.0f * xx4 * xx4;
  return result;
}

void EstepKernel(
  sycl::nd_item<1> &item,
  float *__restrict shared_memory,
  float *__restrict reduce,
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
  float*__restrict  _new_gamma = &shared_memory[0];
  float*__restrict  _phi = &shared_memory[num_topics];
  float*__restrict  _loss_vec = &shared_memory[num_topics * 2];
  float*__restrict  _vali_phi_sum = &shared_memory[num_topics * 3];

  int threadIdx_x = item.get_local_id(0);
  int blockIdx_x = item.get_group(0);
  int blockDim_x = item.get_local_range(0);
  int gridDim_x = item.get_group_range(0);

  float* _grad_alpha = grad_alpha + num_topics * blockIdx_x;

  for (int i = blockIdx_x; i < num_indptr; i += gridDim_x) {
    int beg = indptr[i], end = indptr[i + 1];
    float* _gamma = gamma + num_topics * i;
    if (init_gamma) {
      for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
        _gamma[j] = alpha[j] + (end - beg) / num_topics;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // initiate phi sum for validation data for computing vali loss
    for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
      _vali_phi_sum[j] = 0.0f;

    // iterate E step
    for (int j = 0; j < num_iters; ++j) {
      // initialize new gamma
      for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
        _new_gamma[k] = 0.0f;
      item.barrier(sycl::access::fence_space::local_space);

      // compute phi from gamma
      for (int k = beg; k < end; ++k) {
        const int w = cols[k];  // word
        const bool _vali = vali[k];
        const float c = counts[k];
        // compute phi
        if (not _vali or j + 1 == num_iters) {
          for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
            _phi[l] = beta[w * num_topics + l] * sycl::exp(Digamma(_gamma[l]));
          item.barrier(sycl::access::fence_space::local_space);

          // normalize phi and add it to new gamma and new beta
          float phi_sum = ReduceSum(item, reduce, _phi, num_topics);

          for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
            _phi[l] /= phi_sum;

            // update gamma for train data and phi_sum for computing loss
            if (_vali)
              _vali_phi_sum[l] += _phi[l] * c;
            else
              _new_gamma[l] += _phi[l] * c;

          }
          item.barrier(sycl::access::fence_space::local_space);
        }

        if (j + 1 == num_iters) {
          // update beta for train data
          if (not _vali) {
            // write sycl::access of w th vector of new_beta

            if (threadIdx_x == 0) {
              while (atomicCAS(locks[w], 0, 1)) {}
            }

            item.barrier(sycl::access::fence_space::local_space);
            for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
              new_beta[w * num_topics + l] += _phi[l] * c;
            item.barrier(sycl::access::fence_space::local_space);

            // release lock
            if (threadIdx_x == 0) locks[w] = 0;
            item.barrier(sycl::access::fence_space::local_space);
          }

          // comput loss and reset shared mem
          // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
          for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
            _loss_vec[l] = sycl::log(sycl::fmax(beta[w * num_topics + l], EPS));
            _loss_vec[l] -= sycl::log(sycl::fmax(_phi[l], EPS));
            _loss_vec[l] *= _phi[l];
          }
          item.barrier(sycl::access::fence_space::local_space);
          float _loss = ReduceSum(item, reduce, _loss_vec, num_topics) * c;
          if (threadIdx_x == 0) {
            if (_vali)
              vali_losses[blockIdx_x] += _loss;
            else
              train_losses[blockIdx_x] += _loss;
          }
          item.barrier(sycl::access::fence_space::local_space);

        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      // update gamma
      for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
        _gamma[k] = _new_gamma[k] + alpha[k];
      item.barrier(sycl::access::fence_space::local_space);
    }

    // update gradient of alpha and loss from E[log(theta)]
    float gamma_sum = ReduceSum(item, reduce, _gamma, num_topics);
    for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
      float Elogthetad = Digamma(_gamma[j]) - Digamma(gamma_sum);
      _grad_alpha[j] += Elogthetad;
      _new_gamma[j] *= Elogthetad;
      _vali_phi_sum[j] *= Elogthetad;
    }

    // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
    float train_loss = ReduceSum(item, reduce, _new_gamma, num_topics);
    float vali_loss = ReduceSum(item, reduce, _vali_phi_sum, num_topics);
    if (threadIdx_x == 0) {
      train_losses[blockIdx_x] += train_loss;
      vali_losses[blockIdx_x] += vali_loss;
    }

    item.barrier(sycl::access::fence_space::local_space);
  }
}
