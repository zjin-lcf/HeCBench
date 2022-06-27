/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#pragma omp declare target

//! Thread-local Matrix-Vector multiplication.
template <int n>
void Mv_l(const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = sum;
  }
}

template <int n>
void Mv_l(double alpha, const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = alpha * sum;
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <int n, bool aT = false, bool bT = false>
void MM_l(const double* A, const double* B, double* out)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        double Aik = aT ? A[k + i * n] : A[i + k * n];
        double Bkj = bT ? B[j + k * n] : B[k + j * n];
        sum += Aik * Bkj;
      }
      out[i + j * n] = sum;
    }
  }
}
#pragma omp end declare target

/**
 * Kalman loop kernel. Each thread computes kalman filter for a single series
 * and stores relevant matrices in registers.
 *
 * @tparam     r          Dimension of the state vector
 * @param[in]  ys         Batched time series
 * @param[in]  nobs       Number of observation per series
 * @param[in]  T          Batched transition matrix.            (r x r)
 * @param[in]  Z          Batched "design" vector               (1 x r)
 * @param[in]  RQR        Batched R*Q*R'                        (r x r)
 * @param[in]  P          Batched P                             (r x r)
 * @param[in]  alpha      Batched state vector                  (r x 1)
 * @param[in]  intercept  Do we fit an intercept?
 * @param[in]  d_mu       Batched intercept                     (1)
 * @param[in]  batch_size Batch size
 * @param[out] vs         Batched residuals                     (nobs)
 * @param[out] Fs         Batched variance of prediction errors (nobs)
 * @param[out] sum_logFs  Batched sum of the logs of Fs         (1)
 * @param[in]  n_diff       d + s*D
 * @param[in]  fc_steps   Number of steps to forecast
 * @param[out] d_fc       Array to store the forecast
 * @param[in]  conf_int   Whether to compute confidence intervals
 * @param[out] d_F_fc     Batched variance of forecast errors   (fc_steps)
 */
template <int rd>
void kalman(
  const double*__restrict ys,
  int nobs,
  const double*__restrict T,
  const double*__restrict Z,
  const double*__restrict RQR,
  const double*__restrict P,
  const double*__restrict alpha,
  bool intercept,
  const double*__restrict d_mu,
  int batch_size,
  double*__restrict vs,
  double*__restrict Fs,
  double*__restrict sum_logFs,
  int n_diff,
  int fc_steps = 0,
  double*__restrict d_fc = nullptr,
  bool conf_int = false,
  double* d_F_fc = nullptr)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int bid = 0; bid < batch_size; bid++) {
    constexpr int rd2 = rd * rd;
    double l_RQR[rd2];
    double l_T[rd2];
    double l_Z[rd];
    double l_P[rd2];
    double l_alpha[rd];
    double l_K[rd];
    double l_tmp[rd2];
    double l_TP[rd2];

    // Load global mem into registers
    int b_rd_offset  = bid * rd;
    int b_rd2_offset = bid * rd2;
    for (int i = 0; i < rd2; i++) {
      l_RQR[i] = RQR[b_rd2_offset + i];
      l_T[i]   = T[b_rd2_offset + i];
      l_P[i]   = P[b_rd2_offset + i];
    }
    for (int i = 0; i < rd; i++) {
      if (n_diff > 0) l_Z[i] = Z[b_rd_offset + i];
      l_alpha[i] = alpha[b_rd_offset + i];
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs       = vs + bid * nobs; 
    double* b_Fs       = Fs + bid * nobs;

    double mu = intercept ? d_mu[bid] : 0.0;

    for (int it = 0; it < nobs; it++) {
      // 1. v = y - Z*alpha
      double vs_it = b_ys[it];
      if (n_diff == 0)
        vs_it -= l_alpha[0];
      else {
        for (int i = 0; i < rd; i++) {
          vs_it -= l_alpha[i] * l_Z[i];
        }
      }
      b_vs[it] = vs_it;

      // 2. F = Z*P*Z'
      double _Fs;
      if (n_diff == 0)
        _Fs = l_P[0];
      else {
        _Fs = 0.0;
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
          }
        }
      }
      b_Fs[it] = _Fs;
      if (it >= n_diff) b_sum_logFs += log(_Fs);

      // 3. K = 1/Fs[it] * T*P*Z'
      // TP = T*P
      MM_l<rd>(l_T, l_P, l_TP);
      // K = 1/Fs[it] * TP*Z'
      double _1_Fs = 1.0 / _Fs;
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_K[i] = _1_Fs * l_TP[i];
        }
      } else
        Mv_l<rd>(_1_Fs, l_TP, l_Z, l_K);

      // 4. alpha = T*alpha + K*vs[it] + c
      // tmp = T*alpha
      Mv_l<rd>(l_T, l_alpha, l_tmp);
      // alpha = tmp + K*vs[it]
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i] + l_K[i] * vs_it;
      }
      // alpha = alpha + c
      l_alpha[n_diff] += mu;

      // 5. L = T - K * Z
      // L = T (L is tmp)
      for (int i = 0; i < rd2; i++) {
        l_tmp[i] = l_T[i];
      }
      // L = L - K * Z
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_tmp[i] -= l_K[i];
        }
      } else {
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            l_tmp[j * rd + i] -= l_K[i] * l_Z[j];
          }
        }
      }

      // 6. P = T*P*L' + R*Q*R'
      // P = TP*L'
      MM_l<rd, false, true>(l_TP, l_tmp, l_P);
      // P = P + RQR
      for (int i = 0; i < rd2; i++) {
        l_P[i] += l_RQR[i];
      }
    }
    sum_logFs[bid] = b_sum_logFs;

    // Forecast
    double* b_fc   = fc_steps ? d_fc + bid * fc_steps : nullptr;
    double* b_F_fc = conf_int ? d_F_fc + bid * fc_steps : nullptr;
    for (int it = 0; it < fc_steps; it++) {
      if (n_diff == 0)
        b_fc[it] = l_alpha[0];
      else {
        double pred = 0.0;
        for (int i = 0; i < rd; i++) {
          pred += l_alpha[i] * l_Z[i];
        }
        b_fc[it] = pred;
      }

      // alpha = T*alpha + c
      Mv_l<rd>(l_T, l_alpha, l_tmp);
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i];
      }
      l_alpha[n_diff] += mu;

      if (conf_int) {
        if (n_diff == 0)
          b_F_fc[it] = l_P[0];
        else {
          double _Fs = 0.0;
          for (int i = 0; i < rd; i++) {
            for (int j = 0; j < rd; j++) {
              _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
            }
          }
          b_F_fc[it] = _Fs;
        }

        // P = T*P*T' + RR'
        // TP = T*P
        MM_l<rd>(l_T, l_P, l_TP);
        // P = TP*T'
        MM_l<rd, false, true>(l_TP, l_T, l_P);
        // P = P + RR'
        for (int i = 0; i < rd2; i++) {
          l_P[i] += l_RQR[i];
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <#series> <#observations> <forcast steps> <repeat>\n", argv[0]);
    return 1;
  }
  
  const int nseries = atoi(argv[1]); 
  const int nobs = atoi(argv[2]);
  const int fc_steps = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int rd = 8;
  const int rd2 = rd * rd;
  const int batch_size = nseries;

  const int rd2_word = nseries * rd2;
  const int rd_word = nseries * rd;
  const int nobs_word = nseries * nobs;
  const int ns_word = nseries;
  const int fc_word = fc_steps * nseries;

  const int rd2_size = rd2_word * sizeof(double);
  const int rd_size = rd_word * sizeof(double);
  const int nobs_size = nobs_word * sizeof(double);
  const int ns_size = ns_word * sizeof(double);
  const int fc_size = fc_word * sizeof(double);

  int i;
  srand(123);
  double *RQR = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    RQR[i] = (double)rand() / (double)RAND_MAX;

  double *T = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    T[i] = (double)rand() / (double)RAND_MAX;

  double *P = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    P[i] = (double)rand() / (double)RAND_MAX;

  double *Z = (double*) malloc (rd_size);
  for (i = 0; i < rd * nseries; i++)
    Z[i] = (double)rand() / (double)RAND_MAX;

  double *alpha = (double*) malloc (rd_size);
  for (i = 0; i < rd * nseries; i++)
    alpha[i] = (double)rand() / (double)RAND_MAX;

  double *ys = (double*) malloc (nobs_size);
  for (i = 0; i < nobs * nseries; i++)
    ys[i] = (double)rand() / (double)RAND_MAX;

  double *mu = (double*) malloc (ns_size);
  for (i = 0; i < nseries; i++)
    mu[i] = (double)rand() / (double)RAND_MAX;

  double *vs = (double*) malloc (nobs_size);

  double *Fs = (double*) malloc (nobs_size);

  double *sum_logFs = (double*) malloc (ns_size);

  double *fc = (double*) malloc (fc_size);

  double *F_fc = (double*) malloc (fc_size);

  #pragma omp target data map(to: RQR[0:rd2_word],\
                                    T[0:rd2_word],\
                                    P[0:rd2_word],\
                                    Z[0:rd_word],\
                                    alpha[0:rd_word],\
                                    ys[0:nobs_word],\
                                    mu[0:ns_word]) \
                          map (alloc: vs[0:nobs_word],\
                                    Fs[0:nobs_word],\
                                    sum_logFs[0:ns_word], \
                                    fc[0:fc_word]) \
                          map(from: F_fc[0:fc_word])
  {
    for (int n_diff = 0; n_diff < rd; n_diff++) {

      auto start = std::chrono::steady_clock::now();

      for (i = 0; i < repeat; i++)
        kalman<rd> (
          ys,
          nobs,
          T,
          Z,
          RQR,
          P,
          alpha,
          true, // intercept,
          mu,
          batch_size,
          vs,
          Fs,
          sum_logFs,
          n_diff,
          fc_steps,
          fc,
          true, // forcast
          F_fc );

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time (n_diff = %d): %f (s)\n", n_diff, (time * 1e-9f) / repeat);
    }
  }

  double sum = 0.0;
  for (i = 0; i < fc_steps * nseries - 1; i++)
    sum += (fabs(F_fc[i+1]) - fabs(F_fc[i])) / (fabs(F_fc[i+1]) + fabs(F_fc[i]));
  printf("Checksum: %lf\n", sum);

  free(fc);
  free(F_fc);
  free(sum_logFs);
  free(mu);
  free(Fs);
  free(vs);
  free(ys);
  free(alpha);
  free(Z);
  free(P);
  free(T);
  free(RQR);
  return 0;
}
