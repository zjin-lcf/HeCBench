#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <hip/hip_runtime.h>
#include "kernel.h"

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  // repeat kernel execution
  const int repeat = atoi(argv[1]);

  int i;
  srand(123);

  const int num_topics = 1000;
  const int num_words  = 10266;
  const int block_cnt  = 500;
  const int num_indptr = block_cnt; // max: num_words
  const int block_dim  = 256;
  const int num_iters  = 64;
 
  std::vector<float> alpha(num_topics);
  for (i = 0; i < num_topics; i++)  alpha[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> beta(num_topics * num_words);
  for (i = 0; i < num_topics * num_words; i++)  beta[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> grad_alpha(num_topics * block_cnt, 0.0f);
  std::vector<float> new_beta(num_topics * num_words, 0.0f);
  std::vector<int> h_locks(num_words, 0);
  std::vector<float> gamma (num_indptr * num_topics);

  std::vector<int> indptr (num_indptr+1, 0);
  indptr[num_indptr] = num_words-1;
  for (i = num_indptr; i >= 1; i--) {
    int t = indptr[i] - 1 - (rand() % (num_words/num_indptr));
    if (t < 0) break;
    indptr[i-1] = t;
  }
  const int num_cols = num_words;

  std::vector<int> cols (num_cols);
  std::vector<float> counts (num_cols);

  for (i = 0; i < num_cols; i++) {
    cols[i] = i;
    counts[i] = 0.5f; // arbitrary
  }

  float *d_alpha, *d_beta, *d_grad_alpha, *d_new_beta;
  float *d_counts, *d_gamma, *d_vali_losses, *d_train_losses;
  int *d_locks, *d_cols, *d_indptr;
  bool *d_vali;

  hipMalloc((void**)&d_alpha, sizeof(float) * num_topics);
  hipMemcpy(d_alpha, alpha.data(), sizeof(float) * num_topics, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_beta, sizeof(float) * num_topics * num_words);
  hipMemcpy(d_beta, beta.data(), sizeof(float) * num_topics * num_words, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_grad_alpha, sizeof(float) * num_topics * block_cnt);
  hipMemcpy(d_grad_alpha, grad_alpha.data(), sizeof(float) * block_cnt * num_topics, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_new_beta, sizeof(float) * num_topics * num_words);
  hipMemcpy(d_new_beta, new_beta.data(), sizeof(float) * num_topics * num_words, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_locks, sizeof(int) * num_words);
  hipMemcpy(d_locks, h_locks.data(), sizeof(int) * num_words, hipMemcpyHostToDevice);
  
  hipMalloc((void**)&d_cols, sizeof(int) * num_cols);
  hipMemcpy(d_cols, cols.data(), sizeof(int) * num_cols, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_indptr, sizeof(int) * (num_indptr + 1));
  hipMemcpy(d_indptr, indptr.data(), sizeof(int) * (num_indptr + 1), hipMemcpyHostToDevice);

  hipMalloc((void**)&d_vali, sizeof(bool) * num_cols);

  hipMalloc((void**)&d_counts, sizeof(float) * num_cols);
  hipMemcpy(d_counts, counts.data(), sizeof(float) * num_cols, hipMemcpyHostToDevice);

  // gamma will be initialized in the kernel
  hipMalloc((void**)&d_gamma, sizeof(float) * num_indptr * num_topics);

  // reset losses
  hipMalloc((void**)&d_train_losses, sizeof(float) * block_cnt);
  hipMemset(d_train_losses, 0, sizeof(float) * block_cnt);

  hipMalloc((void**)&d_vali_losses, sizeof(float) * block_cnt);
  hipMemset(d_vali_losses, 0, sizeof(float) * block_cnt);

  // store device results
  std::vector<float> train_losses(block_cnt), vali_losses(block_cnt);

  // training
  hipMemset(d_vali, 0, sizeof(bool) * num_cols); 
  bool init_gamma = false;

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    init_gamma = (i == 0) ? true : false;
    EstepKernel<<<block_cnt, block_dim, 4 * num_topics * sizeof(float)>>>(
      d_cols,
      d_indptr,
      d_vali,
      d_counts,
      init_gamma, num_cols, num_indptr, num_topics, num_iters,
      d_alpha,
      d_beta,
      d_gamma,
      d_grad_alpha,
      d_new_beta,
      d_train_losses,
      d_vali_losses,
      d_locks);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (training): %f (s)\n", (time * 1e-9f) / repeat);

  // validation
  hipMemset(d_vali, 0xFFFFFFFF, sizeof(bool) * num_cols); 

  hipDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    EstepKernel<<<block_cnt, block_dim, 4 * num_topics * sizeof(float)>>>(
      d_cols,
      d_indptr,
      d_vali,
      d_counts,
      init_gamma, num_cols, num_indptr, num_topics, num_iters,
      d_alpha,
      d_beta,
      d_gamma,
      d_grad_alpha,
      d_new_beta,
      d_train_losses,
      d_vali_losses,
      d_locks);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (validation): %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(vali_losses.data(), d_vali_losses, sizeof(float) * block_cnt, hipMemcpyDeviceToHost);
  hipMemcpy(train_losses.data(), d_train_losses, sizeof(float) * block_cnt, hipMemcpyDeviceToHost);

  float total_train_loss = std::accumulate(train_losses.begin(), train_losses.end(), 0.0f);
  float total_vali_loss = std::accumulate(vali_losses.begin(), vali_losses.end(), 0.0f);
  printf("Total train and validate loss: %f %f\n", total_train_loss, total_vali_loss);

  hipFree(d_cols);
  hipFree(d_indptr);
  hipFree(d_vali);
  hipFree(d_counts);
  hipFree(d_alpha);
  hipFree(d_beta);
  hipFree(d_gamma);
  hipFree(d_grad_alpha);
  hipFree(d_new_beta);
  hipFree(d_train_losses);
  hipFree(d_vali_losses);
  hipFree(d_locks);

  return 0;
}
