#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include "common.h"
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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_alpha (alpha.data(), num_topics);
  buffer<float, 1> d_beta (beta.data(), num_topics * num_words);
  buffer<float, 1> d_grad_alpha (grad_alpha.data(), block_cnt * num_topics);
  buffer<float, 1> d_new_beta (new_beta.data(), num_topics * num_words);
  buffer<int, 1> d_locks (h_locks.data(), num_words);
  buffer<int, 1> d_cols (cols.data(), num_cols);
  buffer<int, 1> d_indptr (indptr.data(), num_indptr + 1);
  buffer<bool, 1> d_vali (num_cols);
  buffer<float, 1> d_counts (counts.data(), num_cols);

  // gamma will be initialized in the kernel
  buffer<float, 1> d_gamma (num_indptr * num_topics);

  // reset losses
  buffer<float, 1> d_train_losses (block_cnt);
  q.submit([&] (handler &cgh) {
    auto acc = d_train_losses.get_access<sycl_write>(cgh); 
    cgh.fill(acc, 0.f);
  });

  buffer<float, 1> d_vali_losses (block_cnt);
  q.submit([&] (handler &cgh) {
    auto acc = d_vali_losses.get_access<sycl_write>(cgh); 
    cgh.fill(acc, 0.f);
  });

  // store device results
  std::vector<float> train_losses(block_cnt), vali_losses(block_cnt);

  // training
  q.submit([&] (handler &cgh) {
    auto acc = d_vali.get_access<sycl_write>(cgh); 
    cgh.fill(acc, false);
  });

  range<1> gws (block_cnt * block_dim);
  range<1> lws (block_dim);
  
  bool init_gamma = false;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    init_gamma = (i == 0) ? true : false;
    q.submit([&] (handler &cgh) {
      auto cols = d_cols.get_access<sycl_read>(cgh);
      auto indptr = d_indptr.get_access<sycl_read>(cgh);
      auto vali = d_vali.get_access<sycl_read>(cgh);
      auto counts = d_counts.get_access<sycl_read>(cgh);
      auto alpha = d_alpha.get_access<sycl_read>(cgh);
      auto beta = d_beta.get_access<sycl_read>(cgh);
      auto gamma = d_gamma.get_access<sycl_read_write>(cgh);
      auto grad_alpha = d_grad_alpha.get_access<sycl_read_write>(cgh);
      auto new_beta = d_new_beta.get_access<sycl_read_write>(cgh);
      auto train_losses = d_train_losses.get_access<sycl_read_write>(cgh);
      auto vali_losses = d_vali_losses.get_access<sycl_read_write>(cgh);
      auto locks = d_locks.get_access<sycl_read_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> shared (4 * num_topics, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> reduce (32, cgh);
      cgh.parallel_for<class train_step>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        EstepKernel(
          item,
          shared.get_pointer(), 
          reduce.get_pointer(),
          cols.get_pointer(),
          indptr.get_pointer(),
          vali.get_pointer(),
          counts.get_pointer(),
          init_gamma, num_cols, num_indptr, num_topics, num_iters,
          alpha.get_pointer(),
          beta.get_pointer(),
          gamma.get_pointer(),
          grad_alpha.get_pointer(),
          new_beta.get_pointer(),
          train_losses.get_pointer(),
          vali_losses.get_pointer(),
          locks.get_pointer());
       });
     });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (training): %f (s)\n", (time * 1e-9f) / repeat);

  // validation
  q.submit([&] (handler &cgh) {
    auto acc = d_vali.get_access<sycl_write>(cgh); 
    cgh.fill(acc, true);
  });

  q.wait();
  start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto cols = d_cols.get_access<sycl_read>(cgh);
      auto indptr = d_indptr.get_access<sycl_read>(cgh);
      auto vali = d_vali.get_access<sycl_read>(cgh);
      auto counts = d_counts.get_access<sycl_read>(cgh);
      auto alpha = d_alpha.get_access<sycl_read>(cgh);
      auto beta = d_beta.get_access<sycl_read>(cgh);
      auto gamma = d_gamma.get_access<sycl_read_write>(cgh);
      auto grad_alpha = d_grad_alpha.get_access<sycl_read_write>(cgh);
      auto new_beta = d_new_beta.get_access<sycl_read_write>(cgh);
      auto train_losses = d_train_losses.get_access<sycl_read_write>(cgh);
      auto vali_losses = d_vali_losses.get_access<sycl_read_write>(cgh);
      auto locks = d_locks.get_access<sycl_read_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> shared (4 * num_topics, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> reduce (32, cgh);
      cgh.parallel_for<class vali_step>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        EstepKernel(
          item,
          shared.get_pointer(), 
          reduce.get_pointer(),
          cols.get_pointer(),
          indptr.get_pointer(),
          vali.get_pointer(),
          counts.get_pointer(),
          init_gamma, num_cols, num_indptr, num_topics, num_iters,
          alpha.get_pointer(),
          beta.get_pointer(),
          gamma.get_pointer(),
          grad_alpha.get_pointer(),
          new_beta.get_pointer(),
          train_losses.get_pointer(),
          vali_losses.get_pointer(),
          locks.get_pointer());
       });
     });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (validation): %f (s)\n", (time * 1e-9f) / repeat);

  q.submit([&] (handler &cgh) {
    auto acc = d_vali_losses.get_access<sycl_read>(cgh); 
    cgh.copy(acc, vali_losses.data());
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_train_losses.get_access<sycl_read>(cgh); 
    cgh.copy(acc, train_losses.data());
  });
  q.wait();

  float total_train_loss = std::accumulate(train_losses.begin(), train_losses.end(), 0.0f);
  float total_vali_loss = std::accumulate(vali_losses.begin(), vali_losses.end(), 0.0f);
  printf("Total train and validate loss: %f %f\n", total_train_loss, total_vali_loss);

  return 0;
}
