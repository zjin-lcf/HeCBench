#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define EPS 1e-6f

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

inline float ReduceSum(const float* vec, const int length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++) sum += vec[i];
  return sum;
}

int main() {
  srand(123);
  const int num_topics = 1000;
  const int num_words = 10266;
  const int doc_size = 8;  // Small document like doc 200
  const int num_iters = 64;

  std::vector<float> alpha(num_topics);
  for (int i = 0; i < num_topics; i++)  
    alpha[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> beta(num_topics * num_words);
  for (int i = 0; i < num_topics * num_words; i++)  
    beta[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> gamma(num_topics);
  std::vector<float> _new_gamma(num_topics);
  std::vector<float> _phi(num_topics);
  std::vector<float> _loss_vec(num_topics);

  // Simulate doc 200: 8 words starting at index 7129
  int beg = 7129, end = 7137;
  float train_loss_total = 0.0f;

  // Initialize gamma
  for (int j = 0; j < num_topics; j++) {
    gamma[j] = alpha[j] + (float)(end - beg) / num_topics;
  }

  printf("Initial gamma[0] = %f, doc_size = %d\n", gamma[0], end - beg);

  // E-step iterations
  for (int iter = 0; iter < num_iters; ++iter) {
    for (int k = 0; k < num_topics; k++) _new_gamma[k] = 0.0f;

    for (int k = beg; k < end; ++k) {
      const int w = k;  // Word ID = index
      const float c = 0.5f;

      // Compute phi
      for (int l = 0; l < num_topics; l++)
        _phi[l] = beta[w * num_topics + l] * expf(Digamma(gamma[l]));

      float phi_sum = ReduceSum(_phi.data(), num_topics);
      phi_sum = fmaxf(phi_sum, EPS);

      if (iter == 0 && k == beg) {
        printf("First word: phi_sum = %e, gamma[0] = %f\n", phi_sum, gamma[0]);
      }

      for (int l = 0; l < num_topics; l++) {
        _phi[l] /= phi_sum;
        _new_gamma[l] += _phi[l] * c;
      }

      // Last iteration: compute loss
      if (iter + 1 == num_iters) {
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
      gamma[k] = _new_gamma[k] + alpha[k];
  }

  // Final loss from E[log(theta)]
  float gamma_sum = ReduceSum(gamma.data(), num_topics);
  gamma_sum = fmaxf(gamma_sum, EPS);
  float digamma_sum = Digamma(gamma_sum);
  
  printf("Final gamma[0] = %f, gamma_sum = %f\n", gamma[0], gamma_sum);
  
  for (int j = 0; j < num_topics; j++) {
    float gamma_val = fmaxf(gamma[j], EPS);
    float Elogthetad = Digamma(gamma_val) - digamma_sum;
    _new_gamma[j] *= Elogthetad;
  }

  float train_loss = ReduceSum(_new_gamma.data(), num_topics);
  train_loss_total += train_loss;

  printf("Sequential small doc (size=%d) train loss: %f\n", end-beg, train_loss_total);
  printf("Per-word loss: %f\n", train_loss_total / (end-beg));

  return 0;
}
