#include <cmath>
#include <vector>
#include <stdexcept>

/**
 * Performs the following fused operation for each row (batch element):
 *   1. out[i] = out[i] + input[i] + bias[i % n]
 *   2. mean over the row
 *   3. variance over the row
 *   4. out[i] = ((out[i] - mean) * rsqrt(variance + eps)) * gamma[i] + beta[i]
 *
 * @param out          In/out buffer, shape [num_rows, n].  Acts as residual on
 *                     input and receives the final normalized result.
 * @param input        Residual input,           shape [num_rows, n].
 * @param bias         Bias vector (shared),     shape [n].
 * @param gamma        Layer-norm scale,         shape [n].
 * @param beta         Layer-norm shift,         shape [n].
 * @param layernorm_eps  Small constant for numerical stability.
 * @param num_rows     Number of independent rows (== gridDim.x in the kernel).
 * @param n            Hidden dimension (must be even, matching the /2 in kernel).
 */
template <typename T>
void reference(
          T* out,
    const T* input,
    const T* bias,
    const T* gamma,
    const T* beta,
    float    layernorm_eps,
    int      num_rows,
    int      n)
{
  #pragma omp parallel for
  for (int row = 0; row < num_rows; ++row)
  {
    const int base = row * n;

    float mean = 0.0f;
    #pragma omp parallel for reduction(+:mean)
    for (int i = 0; i < n; ++i)
    {
      out[base + i] += input[base + i] + bias[i];
      mean          += (float)out[base + i];
    }
    mean /= static_cast<float>(n);   // s_mean = mean / n

    float variance = 0.0f;
    #pragma omp parallel for reduction(+:variance)
    for (int i = 0; i < n; ++i)
    {
      float diff  = (float)out[base + i] - mean;
      variance   += diff * diff;
    }
    float inv_std = 1.0f / std::sqrt(variance / static_cast<float>(n) + layernorm_eps);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
      out[base + i] = ((float)out[base + i] - mean) * inv_std * (float)gamma[i] + (float)beta[i];
    }
  }
}
