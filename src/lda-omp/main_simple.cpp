#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "kernel_simple_test.h"

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

  // Allocate on host (will be mapped to device with OpenMP)
  float *d_alpha = alpha.data();
  float *d_beta = beta.data();
  float *d_grad_alpha = grad_alpha.data();
  float *d_new_beta = new_beta.data();
  int *d_locks = h_locks.data();
  int *d_cols = cols.data();
  int *d_indptr = indptr.data();
  float *d_counts = counts.data();

  // Allocate arrays that need initialization
  // Use char instead of bool because std::vector<bool> doesn't have .data()
  std::vector<char> vali_vec(num_cols);
  bool *d_vali = (bool*)vali_vec.data();

  std::vector<float> gamma_vec(num_indptr * num_topics);
  float *d_gamma = gamma_vec.data();

  std::vector<float> train_losses_vec(block_cnt, 0.0f);
  float *d_train_losses = train_losses_vec.data();

  std::vector<float> vali_losses_vec(block_cnt, 0.0f);
  float *d_vali_losses = vali_losses_vec.data();

  // training - set validation flags to false
  std::fill(vali_vec.begin(), vali_vec.end(), false);
  bool init_gamma = false;

  // Map all data to device
  #pragma omp target data map(to: d_cols[0:num_cols], \
                                   d_indptr[0:num_indptr+1], \
                                   d_vali[0:num_cols], \
                                   d_counts[0:num_cols], \
                                   d_alpha[0:num_topics], \
                                   d_beta[0:num_topics*num_words]) \
                          map(tofrom: d_gamma[0:num_indptr*num_topics], \
                                      d_grad_alpha[0:num_topics*block_cnt], \
                                      d_new_beta[0:num_topics*num_words], \
                                      d_train_losses[0:block_cnt], \
                                      d_vali_losses[0:block_cnt], \
                                      d_locks[0:num_words])
  {
    // Explicitly zero out loss arrays on device
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < block_cnt; i++) {
      d_train_losses[i] = 0.0f;
      d_vali_losses[i] = 0.0f;
    }

    auto start = std::chrono::steady_clock::now();

    for (i = 0; i < repeat; i++) {
      init_gamma = (i == 0) ? true : false;
      EstepKernel(
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
        d_locks,
        block_cnt, block_dim);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (training): %f (s)\n", (time * 1e-9f) / repeat);

    // validation - set all validation flags to true
    #pragma omp target update to(d_vali[0:num_cols])
    std::fill(vali_vec.begin(), vali_vec.end(), true);
    #pragma omp target update to(d_vali[0:num_cols])

    start = std::chrono::steady_clock::now();

    for (i = 0; i < repeat; i++) {
      EstepKernel(
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
        d_locks,
        block_cnt, block_dim);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (validation): %f (s)\n", (time * 1e-9f) / repeat);
  } // end target data

  // Debug: check for inf/nan values
  printf("block_cnt = %d\n", block_cnt);
  int inf_count = 0, nan_count = 0, huge_count = 0, zero_count = 0;
  for (int j = 0; j < block_cnt; j++) {
    if (std::isnan(train_losses_vec[j])) nan_count++;
    if (std::isinf(train_losses_vec[j])) inf_count++;
    if (std::abs(train_losses_vec[j]) > 1e6) {
      huge_count++;
      int doc_size = indptr[j+1] - indptr[j];
      printf("DOC %d: HUGE loss = %e, size = %d (beg=%d, end=%d)\n",
             j, train_losses_vec[j], doc_size, indptr[j], indptr[j+1]);
    }
    if (train_losses_vec[j] == 0.0f) {
      zero_count++;
      int doc_size = indptr[j+1] - indptr[j];
      printf("DOC %d: ZERO loss, size = %d (beg=%d, end=%d)\n",
             j, doc_size, indptr[j], indptr[j+1]);
    }
  }
  printf("train_losses: %d nan, %d inf, %d huge (>1e6), %d zero\n",
         nan_count, inf_count, huge_count, zero_count);

  printf("First 10 train_losses: ");
  for (int j = 0; j < 10 && j < block_cnt; j++) {
    printf("%f ", train_losses_vec[j]);
  }
  printf("\n");

  printf("Last 10 train_losses: ");
  for (int j = block_cnt - 10; j < block_cnt; j++) {
    printf("%f ", train_losses_vec[j]);
  }
  printf("\n");

  float total_train_loss = std::accumulate(train_losses_vec.begin(), train_losses_vec.end(), 0.0f);
  float total_vali_loss = std::accumulate(vali_losses_vec.begin(), vali_losses_vec.end(), 0.0f);
  printf("Total train and validate loss: %f %f\n", total_train_loss, total_vali_loss);
  printf("Expected total (if all blocks had -8): %f\n", -8.0f * block_cnt);

  return 0;
}
