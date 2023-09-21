#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <sycl/sycl.hpp>
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_alpha = sycl::malloc_device<float>(num_topics, q);
  q.memcpy(d_alpha, alpha.data(), sizeof(float) * num_topics);

  float *d_beta = sycl::malloc_device<float>(num_topics * num_words, q);
  q.memcpy(d_beta, beta.data(), sizeof(float) * num_topics * num_words);

  float *d_grad_alpha = sycl::malloc_device<float>(block_cnt * num_topics, q);
  q.memcpy(d_grad_alpha, grad_alpha.data(), sizeof(float) * block_cnt * num_topics);

  float *d_new_beta = sycl::malloc_device<float>(num_topics * num_words, q);
  q.memcpy(d_new_beta, new_beta.data(), sizeof(float) * num_topics * num_words);

  int *d_locks = sycl::malloc_device<int>(num_words, q);
  q.memcpy(d_locks, h_locks.data(), sizeof(int) * num_words);

  int *d_cols = sycl::malloc_device<int>(num_cols, q);
  q.memcpy(d_cols, cols.data(), sizeof(int) * num_cols);

  int *d_indptr = sycl::malloc_device<int>(num_indptr + 1, q);
  q.memcpy(d_indptr, indptr.data(), sizeof(int) * (num_indptr + 1));

  bool *d_vali = sycl::malloc_device<bool>(num_cols, q);

  float *d_counts = sycl::malloc_device<float>(num_cols, q);
  q.memcpy(d_counts, counts.data(), sizeof(float) * num_cols);

  // gamma will be initialized in the kernel
  float *d_gamma = sycl::malloc_device<float>(num_indptr * num_topics, q);

  // reset losses
  float *d_train_losses = sycl::malloc_device<float>(block_cnt, q);
  q.memset(d_train_losses, 0, sizeof(float) * block_cnt);

  float *d_vali_losses = sycl::malloc_device<float>(block_cnt, q);
  q.memset(d_vali_losses, 0, sizeof(float) * block_cnt);

  // store device results
  std::vector<float> train_losses(block_cnt), vali_losses(block_cnt);

  // training
  q.memset(d_vali, 0, sizeof(bool) * num_cols);

  sycl::range<1> gws (block_cnt * block_dim);
  sycl::range<1> lws (block_dim);

  bool init_gamma = false;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    init_gamma = (i == 0) ? true : false;
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared (sycl::range<1>(4 * num_topics), cgh);
      sycl::local_accessor<float, 1> reduce (sycl::range<1>(32), cgh);
      cgh.parallel_for<class train_step>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        EstepKernel(
          item,
          shared.get_pointer(),
          reduce.get_pointer(),
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
       });
     });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (training): %f (s)\n", (time * 1e-9f) / repeat);

  // validation
  q.memset(d_vali, true, sizeof(bool) * num_cols);

  q.wait();
  start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared (sycl::range<1>(4 * num_topics), cgh);
      sycl::local_accessor<float, 1> reduce (sycl::range<1>(32), cgh);
      cgh.parallel_for<class vali_step>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        EstepKernel(
          item,
          shared.get_pointer(),
          reduce.get_pointer(),
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
       });
     });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (validation): %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(vali_losses.data(), d_vali_losses, sizeof(float) * block_cnt);
  q.memcpy(train_losses.data(), d_train_losses, sizeof(float) * block_cnt);
  q.wait();

  float total_train_loss = std::accumulate(train_losses.begin(), train_losses.end(), 0.0f);
  float total_vali_loss = std::accumulate(vali_losses.begin(), vali_losses.end(), 0.0f);
  printf("Total train and validate loss: %f %f\n", total_train_loss, total_vali_loss);

  sycl::free(d_cols, q);
  sycl::free(d_indptr, q);
  sycl::free(d_vali, q);
  sycl::free(d_counts, q);
  sycl::free(d_alpha, q);
  sycl::free(d_beta, q);
  sycl::free(d_gamma, q);
  sycl::free(d_grad_alpha, q);
  sycl::free(d_new_beta, q);
  sycl::free(d_train_losses, q);
  sycl::free(d_vali_losses, q);
  sycl::free(d_locks, q);
  return 0;
}
