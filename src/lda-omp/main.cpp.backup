#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <math.h>
#include <omp.h>
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

  float *d_alpha = alpha.data(); 
  float *d_beta = beta.data(); 
  float *d_grad_alpha = grad_alpha.data();
  float *d_new_beta = new_beta.data();
  float *d_counts = counts.data();
  int *d_locks = h_locks.data();
  int *d_cols = cols.data();
  int *d_indptr = indptr.data();

  // reset the values
  bool *d_vali = (bool*) calloc (num_cols, sizeof(bool));

  // gamma will be initialized in the kernel
  float *d_gamma = (float*) malloc (sizeof(float) * num_indptr * num_topics);

  // store device results (reset losses)
  std::vector<float> train_losses(block_cnt, 0.f), vali_losses(block_cnt, 0.f);

  float *d_train_losses = train_losses.data();
  float *d_vali_losses = vali_losses.data();

  #pragma omp target data map (to: d_alpha[0:num_topics], \
                                   d_beta[0:num_topics * num_words],\
                                   d_grad_alpha[0:num_topics * block_cnt],\
                                   d_new_beta[0:num_topics * num_words],\
                                   d_locks[0:num_words],\
                                   d_cols[0:num_cols],\
                                   d_indptr[0:num_indptr+1],\
                                   d_counts[0:num_cols],\
                                   d_vali[0:num_cols]) \
                          map (tofrom: d_train_losses[0:block_cnt], \
                                       d_vali_losses[0:block_cnt]) \
                          map (alloc: d_gamma[0:num_indptr * num_topics])
  {
    // training
    bool init_gamma = false;

    auto start = std::chrono::steady_clock::now();

    for (i = 0; i < repeat; i++) {
      init_gamma = (i == 0) ? true : false;
      EstepKernel<block_cnt, block_dim, 4 * num_topics>(
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

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (training): %f (s)\n", (time * 1e-9f) / repeat);

    // validation
    memset(d_vali, 0xFFFFFFFF, sizeof(bool) * num_cols); 
    #pragma omp target update to (d_vali[0:num_cols])

    start = std::chrono::steady_clock::now();

    for (i = 0; i < repeat; i++) {
      EstepKernel<block_cnt, block_dim, 4 * num_topics>(
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

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (validation): %f (s)\n", (time * 1e-9f) / repeat);
  }

  float total_train_loss = std::accumulate(train_losses.begin(), train_losses.end(), 0.0f);
  float total_vali_loss = std::accumulate(vali_losses.begin(), vali_losses.end(), 0.0f);
  printf("Total train and validate loss: %f %f\n", total_train_loss, total_vali_loss);

  free(d_vali);
  free(d_gamma);

  return 0;
}
