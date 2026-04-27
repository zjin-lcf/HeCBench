#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
#include "reference.h"

void mha_omp(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const int beam_size,
    const int n_steps,
    const int qk_col,
    const int v_col,
    const int nhead,
    const float scale,
    const int THRESHOLD,
    float * __restrict__ dst,
    int block_size)
{
  int i, t;
  int dim_per_head = qk_col / nhead;
  #pragma omp target teams distribute collapse(2)
  for (int candidate_id = 0; candidate_id < beam_size; candidate_id++) {
    for (int head_id = 0; head_id < nhead; head_id++) {
      //float buffer[qk_col / nhead + n_steps];
      float buffer[512];
      float *sq = buffer;
      float *logits = buffer + dim_per_head;
      #pragma omp parallel for num_threads(block_size)
      for (t = 0; t < dim_per_head; t++) {
        int pos = candidate_id * qk_col + head_id * dim_per_head + t;
        sq[t] = q[pos];
      }
      float dp[n_steps];
      #pragma omp parallel for num_threads(block_size)
      for (t = 0; t < n_steps; t++) {
        const float *k2 = k + candidate_id * qk_col * n_steps + head_id * dim_per_head + t * qk_col;
        float summ = 0.f;
        #pragma omp parallel for reduction(+:summ)
        for (i = 0; i < dim_per_head; i++)
          summ += sq[i] * k2[i];
        summ *= scale;
        dp[t] = summ;
      }
      float max_val = -1e-20f;
      #pragma omp parallel for reduction(max:max_val) num_threads(block_size)
      for (t = 0; t < n_steps; t++) {
        max_val = fmaxf(max_val, dp[t]);
      }
      float val = 0;
      #pragma omp parallel for reduction(+:val) num_threads(block_size)
      for (t = 0; t < n_steps; t++) {
        dp[t] -= max_val;
        if(dp[t] < -THRESHOLD) dp[t] = -THRESHOLD;
        val += expf(dp[t]);
      }
      #pragma omp parallel for num_threads(block_size)
      for (t = 0; t < n_steps; t++) {
        logits[t] = expf(dp[t]) / val;
      }
      #pragma omp parallel for num_threads(block_size)
      for (t = 0; t < dim_per_head; t++) {
        int tid = candidate_id * v_col * n_steps + head_id * dim_per_head + t;
        float summ = 0.f;
        #pragma omp parallel for reduction(+:summ)
        for (i = 0; i < n_steps; ++i)
          summ += logits[i] * v[tid + i * v_col];
        dst[candidate_id * v_col + head_id * dim_per_head + t] = summ;
      }
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int beamsize    = 4;
  const int nhead       = 16;
  const int dim_feature = nhead * 256;
  const int n_steps     = 9;

  const float scaler = sqrtf(nhead * 1.f / dim_feature);

  const int qk_col    = dim_feature;
  const int v_col     = dim_feature;
  const int THRESHOLD = 64;

  const int q_size = beamsize * dim_feature;
  const int k_size = beamsize * dim_feature * n_steps;
  const int v_size = beamsize * dim_feature * n_steps;

  float *hq    = (float *)malloc(sizeof(float) * q_size);
  float *hk    = (float *)malloc(sizeof(float) * k_size);
  float *hv    = (float *)malloc(sizeof(float) * v_size);
  float *h_dst = (float *)malloc(sizeof(float) * q_size);
  float *r_dst = (float *)malloc(sizeof(float) * q_size);

  srand(123);
  for (int i = 0; i < q_size; i++) hq[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < k_size; i++) hk[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < v_size; i++) hv[i] = rand() / (float)RAND_MAX;

  int block_size = qk_col / nhead;

  #pragma omp target data map(to: hq[0:q_size], hk[0:k_size], hv[0:v_size])  \
                          map(from: h_dst[0:q_size])
  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      mha_omp(hq, hk, hv, beamsize, n_steps, qk_col, v_col, nhead, scaler,
              THRESHOLD, h_dst, block_size);
    }
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
  }
  // reference check
  mha_reference(hq, hk, hv, beamsize, n_steps, qk_col, v_col, nhead, scaler, THRESHOLD, r_dst);

  bool ok = true;
  for (int i = 0; i < beamsize && ok; i++) {
    for (int j = 0; j < dim_feature; j++) {
      if (fabsf(h_dst[i * dim_feature + j] - r_dst[i * dim_feature + j]) > 1e-3f) {
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(hq);
  free(hk);
  free(hv);
  free(h_dst);
  free(r_dst);
  return 0;
}

