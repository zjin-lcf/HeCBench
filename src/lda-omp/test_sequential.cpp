#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#define EPS 1e-6f

// Digamma function
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

// Simple reduction
inline float ReduceSum(const float* vec, const int length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++)
    sum += vec[i];
  return sum;
}

int main() {
  int i;
  srand(123);

  const int num_topics = 1000;
  const int num_words  = 10266;
  const int num_indptr = 1; // Just process ONE document
  const int num_iters  = 64;
 
  std::vector<float> alpha(num_topics);
  for (i = 0; i < num_topics; i++)  alpha[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> beta(num_topics * num_words);
  for (i = 0; i < num_topics * num_words; i++)  beta[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> gamma(num_indptr * num_topics);

  std::vector<int> indptr(num_indptr+1, 0);
  indptr[num_indptr] = num_words-1;
  
  const int num_cols = num_words;

  std::vector<int> cols(num_cols);
  std::vector<float> counts(num_cols);

  for (i = 0; i < num_cols; i++) {
    cols[i] = i;
    counts[i] = 0.5f;
  }

  std::vector<bool> vali(num_cols, false);

  // Allocate shared memory (simulate what GPU does)
  std::vector<float> _new_gamma(num_topics);
  std::vector<float> _phi(num_topics);
  std::vector<float> _loss_vec(num_topics);
  std::vector<float> _vali_phi_sum(num_topics);

  float train_loss_total = 0.0f;

  // Process one document (simulate one team/block)
  for (int doc_id = 0; doc_id < num_indptr; doc_id++) {
    int beg = indptr[doc_id], end = indptr[doc_id + 1];
    float* _gamma = &gamma[num_topics * doc_id];

    // Initialize gamma
    for (int j = 0; j < num_topics; j++) {
      _gamma[j] = alpha[j] + (float)(end - beg) / num_topics;
    }

    // Initialize validation phi sum
    for (int j = 0; j < num_topics; j++)
      _vali_phi_sum[j] = 0.0f;

    // E-step iterations
    for (int iter = 0; iter < num_iters; ++iter) {
      // Initialize new gamma
      for (int k = 0; k < num_topics; k++)
        _new_gamma[k] = 0.0f;

      // Process each word
      for (int k = beg; k < end; ++k) {
        const int w = cols[k];
        const bool _vali = vali[k];
        const float c = counts[k];

        // Compute phi
        if (!_vali || iter + 1 == num_iters) {
          for (int l = 0; l < num_topics; l++)
            _phi[l] = beta[w * num_topics + l] * expf(Digamma(_gamma[l]));

          // Normalize phi
          float phi_sum = ReduceSum(_phi.data(), num_topics);
          phi_sum = fmaxf(phi_sum, EPS);

          for (int l = 0; l < num_topics; l++) {
            _phi[l] /= phi_sum;

            if (_vali)
              _vali_phi_sum[l] += _phi[l] * c;
            else
              _new_gamma[l] += _phi[l] * c;
          }
        }

        // Last iteration: compute loss
        if (iter + 1 == num_iters && !_vali) {
          for (int l = 0; l < num_topics; l++) {
            float beta_val = fmaxf(beta[w * num_topics + l], EPS);
            float phi_val = fmaxf(_phi[l], EPS);
            beta_val = fminf(beta_val, 1.0f);
            phi_val = fminf(phi_val, 1.0f);
            _loss_vec[l] = logf(beta_val) - logf(phi_val);
            _loss_vec[l] *= phi_val;
          }

          float _loss = ReduceSum(_loss_vec.data(), num_topics) * c;
          train_loss_total += _loss;
        }
      }

      // Update gamma
      for (int k = 0; k < num_topics; k++)
        _gamma[k] = _new_gamma[k] + alpha[k];
    }

    // Update final loss from E[log(theta)]
    float gamma_sum = ReduceSum(_gamma, num_topics);
    gamma_sum = fmaxf(gamma_sum, EPS);
    float digamma_sum = Digamma(gamma_sum);
    for (int j = 0; j < num_topics; j++) {
      float gamma_val = fmaxf(_gamma[j], EPS);
      float Elogthetad = Digamma(gamma_val) - digamma_sum;
      _new_gamma[j] *= Elogthetad;
    }

    float train_loss = ReduceSum(_new_gamma.data(), num_topics);
    train_loss_total += train_loss;
  }

  printf("Sequential single-document train loss: %f\n", train_loss_total);
  printf("Expected: around -8 to -10 for one document\n");

  return 0;
}
