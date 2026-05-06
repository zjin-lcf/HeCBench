// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "kernels.h"
#include "reference.h"

void launch(
    int *block_count,            // [BATCH, N_HEADS, NUM_ROWS]
    int *block_offset,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_S]
    int *column_count,           // [BATCH, N_HEADS, NUM_ROWS]
    int *column_index,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    int *q_seqlens,               // [BATCH, ]
    int *kv_seqlens,              // [BATCH, ]
    int *vertical_indexes,        // [BATCH, N_HEADS, NNZ_V]
    int *slash_indexes,           // [BATCH, N_HEADS, NNZ_S]
    int *vertical_indices_count,  // [N_HEADS, ]
    int *slash_indices_count,
    int64_t context_size,
    int64_t block_size_M,
    int64_t block_size_N,
    bool causal,
    int batch_size, // = slash_indexes.size(0);
    int num_heads, // = slash_indexes.size(1);
    int nnz_slash, // = slash_indexes.size(2);
    int nnz_vertical) // = vertical_indexes.size(2);
{
  const int num_rows = (context_size + block_size_M - 1) / block_size_M;

  convert_vertical_slash_indexes_64x64_mergehead(
      q_seqlens,
      kv_seqlens,
      vertical_indexes,
      slash_indexes,
      vertical_indices_count,
      slash_indices_count,
      block_count,
      block_offset,
      column_count,
      column_index,
      batch_size,
      num_heads,
      num_rows,
      block_size_M,
      block_size_N,
      nnz_vertical,
      nnz_slash,
      causal);
}

#define GPU_CHECK(call)                                           \
    do {                                                          \
        hipError_t _e = (call);                                   \
        if (_e != hipSuccess) {                                   \
            fprintf(stderr, "HIP error %s:%d  %s\n",              \
                    __FILE__, __LINE__, hipGetErrorString(_e));   \
        }                                                         \
    } while (0)

struct MergeheadResult {
    int* block_count;    // device  [B, H, num_rows]
    int* block_offset;   // device  [B, H, num_rows, NNZ_S]
    int* column_count;   // device  [B, H, num_rows]
    int* column_index;   // device  [B, H, num_rows, NNZ_V]
    int n_scalar;        // B * H * num_rows
    int n_offset;        // B * H * num_rows * NNZ_S
    int n_index;         // B * H * num_rows * NNZ_V
    int num_rows, NNZ_V, NNZ_S;
};

void init_input(int* h_vidx, int* h_sidx, int* h_vcnt, int* h_scnt,
                int BATCH, int N_HEADS, int NNZ_V, int NNZ_S,
                int num_cols, int num_rows,
                int n_sinks = 2, int window_diags = 4)
{
  int eff_v = (n_sinks < NNZ_V) ? n_sinks : NNZ_V;
  int eff_s = (window_diags < NNZ_S) ? window_diags : NNZ_S;

  for (int b = 0; b < BATCH; ++b) {
    for (int h = 0; h < N_HEADS; ++h) {
      int base_v = (b * N_HEADS + h) * NNZ_V;
      int base_s = (b * N_HEADS + h) * NNZ_S;

      for (int i = 0; i < eff_v; ++i)
        h_vidx[base_v + i] = i;

      for (int i = 0; i < eff_s; ++i)
        h_sidx[base_s + i] = num_rows - 1 + i;
    }
  }

  for (int h = 0; h < N_HEADS; ++h) {
    h_vcnt[h] = eff_v;
    h_scnt[h] = eff_s;
  }
}

struct configs {
  int batch, n_heads, nnz_v, nnz_s, context_size, block_size;
  const char* label;
} cases[] = {
  {1,   8,  16,  16,   4096, 64, "case1"},
  {1,  32,  32,  32, 131072, 128, "case2"},
  {4,  64,  64,  64,  16384, 64, "case3"},
  {8,  32,  64,  64,  32768, 64, "case4"},
  {16, 64, 256, 256,  65536, 64, "case5"},
};

void run_test(int repeat, bool causal, struct configs &cfg)
{
  const int BATCH = cfg.batch;
  const int N_HEADS = cfg.n_heads;
  const int NNZ_V = cfg.nnz_v;
  const int NNZ_S = cfg.nnz_s;
  const int context_size = cfg.context_size;
  const int block_size_M = cfg.block_size,
            block_size_N = cfg.block_size;
  printf("batch %4d, #heads %3d, nnz_v %3d, nnz_s %3d, context size %6d, block size %3d, causal %d\n",
         BATCH, N_HEADS, NNZ_V, NNZ_S, context_size, block_size_M, causal);

  const int num_rows = (context_size+block_size_M-1)/block_size_M;
  const int num_cols = (context_size+block_size_N-1)/block_size_N;

  // q/kv array size
  size_t qkv_size = BATCH;
  // veritical index 
  size_t vidx_size = BATCH * N_HEADS * NNZ_V;
  // slash index 
  size_t sidx_size = BATCH * N_HEADS * NNZ_S;
  // counter
  size_t idxcnt_size = N_HEADS;

  int *h_q = (int*)malloc(BATCH * sizeof(int));
  int *h_kv = (int*)malloc(BATCH * sizeof(int));

  srand(131072);
  for (int i = 0; i < BATCH; i++) {
    h_q[i] = h_kv[i] = (1 + rand() % (context_size / block_size_M)) * block_size_M;
  }

  // vertical indices count
  int *h_vidx = (int*)malloc(vidx_size * sizeof(int));
  // slash indices count
  int *h_sidx = (int*)malloc(sidx_size * sizeof(int));
  // per head 
  int *h_vcnt = (int*)malloc(N_HEADS * sizeof(int));
  int *h_scnt = (int*)malloc(N_HEADS * sizeof(int));

  init_input(h_vidx, h_sidx, h_vcnt, h_scnt,
             BATCH, N_HEADS, NNZ_V, NNZ_S, num_cols, num_rows);

  MergeheadResult res;
  res.num_rows = num_rows; res.NNZ_V = NNZ_V; res.NNZ_S = NNZ_S;
  res.n_scalar = BATCH * N_HEADS * num_rows;
  res.n_offset = BATCH * N_HEADS * num_rows * NNZ_S;
  res.n_index  = BATCH * N_HEADS * num_rows * NNZ_V;

  int *h_col_cnt = (int*)malloc(res.n_scalar * sizeof(int));
  int *h_blk_cnt = (int*)malloc(res.n_scalar * sizeof(int));
  int *h_col_idx = (int*)malloc(res.n_index * sizeof(int));
  int *h_blk_off = (int*)malloc(res.n_offset * sizeof(int));

  int *r_col_cnt = (int*)calloc(res.n_scalar, sizeof(int));
  int *r_blk_cnt = (int*)calloc(res.n_scalar, sizeof(int));
  int *r_col_idx = (int*)calloc(res.n_index, sizeof(int));
  int *r_blk_off = (int*)calloc(res.n_offset, sizeof(int));

  reference (
    h_q,
    h_kv,
    h_vidx,
    h_sidx,
    h_vcnt, // per_head_vertical_topkv 
    h_scnt, // per_head_slash_topkv
    r_blk_cnt, //block_count,
    r_blk_off, //block_offset,
    r_col_cnt, //column_count,
    r_col_idx, //column_index,
    BATCH,     //batch_size,
    N_HEADS,   //num_heads,
    num_rows,
    block_size_M,
    block_size_N,
    NNZ_V, //nnz_vertical,
    NNZ_S, //nnz_slash,
    causal);

  int *d_q, *d_kv, *d_vidx, *d_sidx, *d_vcnt, *d_scnt;
  GPU_CHECK(hipMalloc(&d_q,    qkv_size * sizeof(int)));
  GPU_CHECK(hipMalloc(&d_kv,   qkv_size * sizeof(int)));
  GPU_CHECK(hipMalloc(&d_vidx, vidx_size * sizeof(int)));
  GPU_CHECK(hipMalloc(&d_sidx, sidx_size * sizeof(int)));
  GPU_CHECK(hipMalloc(&d_vcnt, idxcnt_size * sizeof(int)));
  GPU_CHECK(hipMalloc(&d_scnt, idxcnt_size * sizeof(int)));

  GPU_CHECK(hipMemcpy(d_q, h_q, qkv_size * sizeof(int), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_kv, h_kv, qkv_size * sizeof(int), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_vidx, h_vidx, vidx_size * sizeof(int), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_sidx, h_sidx, sidx_size * sizeof(int), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_vcnt, h_vcnt, idxcnt_size * sizeof(int), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_scnt, h_scnt, idxcnt_size * sizeof(int), hipMemcpyHostToDevice));

  GPU_CHECK(hipMalloc(&res.block_count,  res.n_scalar * sizeof(int)));
  GPU_CHECK(hipMemset(res.block_count, 0, res.n_scalar * sizeof(int)));

  GPU_CHECK(hipMalloc(&res.column_count, res.n_scalar * sizeof(int)));
  GPU_CHECK(hipMemset(res.column_count, 0, res.n_scalar * sizeof(int)));

  GPU_CHECK(hipMalloc(&res.block_offset, res.n_offset * sizeof(int)));
  GPU_CHECK(hipMemset(res.block_offset, 0, res.n_offset * sizeof(int)));

  GPU_CHECK(hipMalloc(&res.column_index, res.n_index  * sizeof(int)));
  GPU_CHECK(hipMemset(res.column_index, 0, res.n_index * sizeof(int)));

  GPU_CHECK(hipDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    launch(res.block_count, res.block_offset,
           res.column_count, res.column_index,
           d_q, d_kv,
           d_vidx, d_sidx,
           d_vcnt, d_scnt,
           context_size, block_size_M, block_size_N,
           causal, BATCH, N_HEADS, NNZ_S, NNZ_V);
  }
  GPU_CHECK(hipDeviceSynchronize());

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  GPU_CHECK(hipMemcpy(h_blk_cnt, res.block_count,  res.n_scalar*sizeof(int), hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(h_col_cnt, res.column_count, res.n_scalar*sizeof(int), hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(h_blk_off, res.block_offset, res.n_offset*sizeof(int), hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(h_col_idx, res.column_index, res.n_index *sizeof(int), hipMemcpyDeviceToHost));

  #ifdef DEBUG
  for (int i = 0; i < res.n_scalar; i++) {
    if (h_col_cnt[i] != r_col_cnt[i]) {
      printf("col cnt @%d: %d %d\n", i, h_col_cnt[i], r_col_cnt[i]);
      break;
    }
  }
  for (int i = 0; i < res.n_scalar; i++) {
    if (h_blk_cnt[i] != r_blk_cnt[i]) {
      printf("blk cnt @%d: %d %d\n", i, h_blk_cnt[i], r_blk_cnt[i]);
      break;
    }
  }
  for (int i = 0; i < res.n_index; i++) {
    if (h_col_idx[i] != r_col_idx[i]) {
      printf("col idx @%d: %d %d\n", i, h_col_idx[i], r_col_idx[i]);
      break;
    }
  }
  for (int i = 0; i < res.n_offset; i++) {
    if (h_blk_off[i] != r_blk_off[i]) {
      printf("blk off @%d: %d %d\n", i, h_blk_off[i], r_blk_off[i]);
      break;
    }
  }
  #else
  int status = 0;
  status += memcmp(h_col_cnt, r_col_cnt, res.n_scalar*sizeof(int));
  status += memcmp(h_blk_cnt, r_blk_cnt, res.n_scalar*sizeof(int));
  status += memcmp(h_col_idx, r_col_idx, res.n_index*sizeof(int));
  status += memcmp(h_blk_off, r_blk_off, res.n_offset*sizeof(int));
  printf("%s\n", status ? "FAIL" : "PASS"); 
  #endif

  GPU_CHECK(hipFree(d_q)); GPU_CHECK(hipFree(d_kv));
  GPU_CHECK(hipFree(d_vidx)); GPU_CHECK(hipFree(d_sidx));
  GPU_CHECK(hipFree(d_vcnt)); GPU_CHECK(hipFree(d_scnt));
  GPU_CHECK(hipFree(res.block_count)); GPU_CHECK(hipFree(res.block_offset));
  GPU_CHECK(hipFree(res.column_count)); GPU_CHECK(hipFree(res.column_index));

  free(h_q); free(h_kv); free(h_vidx); free(h_sidx);
  free(h_vcnt); free(h_scnt);
  free(h_col_cnt); free(h_blk_cnt); free(h_col_idx); free(h_blk_off);
  free(r_col_cnt); free(r_blk_cnt); free(r_col_idx); free(r_blk_off);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  for (int i = 0; i < 5; i++) {
    run_test(repeat, true, cases[i]);
    run_test(repeat, false, cases[i]);
  }
  printf("----- warmup done -----\n");

  for (int i = 0; i < 5; i++) {
    run_test(repeat, true, cases[i]);
    run_test(repeat, false, cases[i]);
  }
  return 0;
}
