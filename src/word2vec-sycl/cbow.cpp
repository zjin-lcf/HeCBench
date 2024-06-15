#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "cbow.h"

extern real *syn0;
extern int * table;
extern int vocab_size, layer1_size , layer1_size_aligned;
extern int negative , window;
extern int table_size;
// To batch data to minimize data transfer, sen stores words + alpha values
// alpha value start at offset = MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH

extern int * sen;

real * d_expTable = NULL;
real * d_syn0 = NULL;
real * d_syn1neg = NULL;
int  * d_sen = NULL;
unsigned int * d_random = NULL;
int * d_table = NULL;

int maxThreadsPerBlock = 256;
int numBlock;
int shared_mem_usage;


void device_memset(real * array, int size, const sycl::nd_item<3> &item){
  int idx = item.get_global_id(2);
  if (idx < size)
    array[idx] = 0;
}

void reduceInWarp(volatile float * f, int idInWarp,
                  const sycl::nd_item<3> &item){

  for (unsigned int i=THREADS_PER_WORD /2; i>32; i>>=1) {
    if (idInWarp < i) {
      f[idInWarp] += f[idInWarp + i];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
  if (idInWarp < 32){
    f[idInWarp] += f[idInWarp + 32];
    f[idInWarp] += f[idInWarp + 16];
    f[idInWarp] += f[idInWarp + 8];
    f[idInWarp] += f[idInWarp + 4];
    f[idInWarp] += f[idInWarp + 2];
    f[idInWarp] += f[idInWarp + 1];
  }
}


void device_cbow(
    const int sentence_num,
    const int layer1_size,
    const int layer1_size_aligned,
    const int window,
    const int negative,
    const int table_size,
    const int vocab_size,
    const int *__restrict__ d_sen,
    const int *__restrict__ d_table,
    float *__restrict__ d_syn0,
    float *__restrict__ d_syn1neg,
    unsigned int *__restrict__ d_random,
    const sycl::nd_item<3> &item,
    real *shared,
    real const *expTable)
{
  int sentence_position =
      (item.get_local_id(2) / THREADS_PER_WORD) +
      (item.get_local_range(2) / THREADS_PER_WORD) * item.get_group(2);
  int idInWarp = item.get_local_id(2) % THREADS_PER_WORD;

  float *f =
      shared + (item.get_local_id(2) / THREADS_PER_WORD) * THREADS_PER_WORD;
  float *neu1 =
      shared + BLOCK_SIZE +
      (item.get_local_id(2) / THREADS_PER_WORD) * layer1_size_aligned;
  float *neu1e =
      shared + BLOCK_SIZE +
      (item.get_local_range(2) / THREADS_PER_WORD) * layer1_size_aligned +
      (item.get_local_id(2) / THREADS_PER_WORD) * layer1_size_aligned;

  if (sentence_position < MAX_SENTENCE_LENGTH) {
    unsigned int next_random = d_random[sentence_position];

    for (int sentence_idx = 0; sentence_idx < sentence_num; sentence_idx++) {

      for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD) {
        neu1[c] = 0;
        neu1e[c] = 0;
      }

      next_random = next_random * (unsigned int) 1664525 + 1013904223;
      int b = next_random % window;
      int word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + sentence_position];
      // in -> hidden
      int cw = 0;
      for (int a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          int w = sentence_position - window + a;
          if (w < 0 || w>= MAX_SENTENCE_LENGTH) continue;
          int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];
          for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
            neu1[c] += d_syn0[c + last_word * layer1_size_aligned];

          cw++;
        }

      if (cw) {
        for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
          neu1[c] /= cw;

        // NEGATIVE SAMPLING
        int target, label;
        float alpha =*((float *) &d_sen[MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + sentence_idx]);

        if (negative > 0)

          for (int d = 0; d < negative + 1; d++) {

            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned int) 1664525 + 1013904223;
              target = d_table[(next_random) % table_size];
              if (target == 0)
                target = next_random % (vocab_size - 1) + 1;
              if (target == word)
                continue;
              label = 0;
            }
            int l2 = target * layer1_size_aligned;
            f[idInWarp] = 0;


            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD){
              f[idInWarp] += neu1[c] * d_syn1neg[c + l2];   
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Do reduction here;
            reduceInWarp(f, idInWarp, item);

            item.barrier(sycl::access::fence_space::local_space);

            float g;
            if (f[0] > MAX_EXP)
              g = (label - 1) * alpha;
            else if (f[0] < -MAX_EXP)
              g = (label - 0) * alpha;
            else
              g = (label - expTable[(int) ((f[0] + MAX_EXP)
                    * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
              neu1e[c] += g * d_syn1neg[c + l2];
            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
              d_syn1neg[c + l2] += g * neu1[c];
          }

        // hidden -> in
        for (int a = b; a < window * 2 + 1 - b; a++)
          if (a != window) {
            int w = sentence_position - window + a;
            if (w < 0)
              continue;
            if (w >= MAX_SENTENCE_LENGTH)
              continue;
            int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];

            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
              d_syn0[c + last_word * layer1_size_aligned] += neu1e[c];
          }
      }
    }// End for sentence_idx

    // Update d_random
    if (idInWarp == 0 ) d_random[sentence_position] = next_random;
  }
}

void cleanUpGPU(sycl::queue &q) {
  sycl::free(d_syn1neg, q);
  sycl::free(d_syn0, q);
  sycl::free(sen, q);
  sycl::free(d_sen, q);
  sycl::free(d_random, q);
  sycl::free(d_table, q);
  sycl::free(d_expTable, q);
}

void initializeGPU(sycl::queue &q) {
  real * h_expTable = (real *)malloc((EXP_TABLE_SIZE ) * sizeof(real));
  for (int i = 0; i < EXP_TABLE_SIZE; i++) {
    h_expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    h_expTable[i] = h_expTable[i] / (h_expTable[i] + 1);
  }

  d_expTable = sycl::malloc_device<real>(EXP_TABLE_SIZE, q);
  q.memcpy(d_expTable, h_expTable, sizeof(real) * EXP_TABLE_SIZE).wait();
  free(h_expTable);

  if (negative>0) {
    int syn1neg_size = vocab_size * layer1_size_aligned;
    d_syn1neg = sycl::malloc_device<real>(syn1neg_size, q);
    q.submit([&](sycl::handler &cgh) {
      real *d_syn1neg_p = d_syn1neg;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, syn1neg_size / maxThreadsPerBlock + 1) *
                  sycl::range<3>(1, 1, maxThreadsPerBlock),
              sycl::range<3>(1, 1, maxThreadsPerBlock)),
          [=](sycl::nd_item<3> item) {
            device_memset(d_syn1neg_p, syn1neg_size, item);
          });
    }).wait();
  }

  int syn0_size = vocab_size * layer1_size_aligned;
  d_syn0 = sycl::malloc_device<real>(syn0_size, q);
  q.memcpy(d_syn0, syn0, syn0_size * sizeof(real)).wait();

  sen = sycl::malloc_host<int>((MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM), q);
  d_sen = sycl::malloc_device<int>((MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM), q);
  d_random = sycl::malloc_device<unsigned int>(MAX_SENTENCE_LENGTH, q);

  int h_random[MAX_SENTENCE_LENGTH];

  srand(123);
  for (int i = 0 ; i < MAX_SENTENCE_LENGTH; i++)
    h_random[i] = (unsigned int) rand();

  q.memcpy(d_random, h_random, MAX_SENTENCE_LENGTH * sizeof(unsigned int)).wait();
  d_table = sycl::malloc_device<int>(table_size, q);
  q.memcpy(d_table, table, table_size * sizeof(int)).wait();

  numBlock = MAX_SENTENCE_LENGTH / (BLOCK_SIZE/THREADS_PER_WORD) + 1;
  shared_mem_usage = (BLOCK_SIZE + (BLOCK_SIZE/THREADS_PER_WORD) * layer1_size_aligned * 2);
}

void TransferDataToGPU(sycl::queue &q) {
  q.memcpy(d_sen, sen, (MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM) * sizeof(int)).wait();
}

void GetResultFromGPU(sycl::queue &q) {
  q.memcpy(syn0, d_syn0, vocab_size * layer1_size_aligned * sizeof(real)).wait();
}

void TrainGPU(sycl::queue &q, int sentence_num) {
  TransferDataToGPU(q);

  q.submit([&](sycl::handler &cgh) {

    sycl::local_accessor<real, 1> sm (
        sycl::range<1>(shared_mem_usage), cgh);

    auto expTable_p = d_expTable;
    const int layer1_size_p = layer1_size;
    const int layer1_size_aligned_p = layer1_size_aligned;
    const int window_p = window;
    const int negative_p = negative;
    const int table_size_p = table_size;
    const int vocab_size_p = vocab_size;
    const int *d_sen_p = d_sen;
    const int *d_table_p = d_table;
    real *d_syn0_p = d_syn0;
    real *d_syn1neg_p = d_syn1neg;
    unsigned int *d_random_p = d_random;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, numBlock) *
                          sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item) {
          device_cbow(
              sentence_num, layer1_size_p, layer1_size_aligned_p,
              window_p, negative_p, table_size_p, vocab_size_p,
              d_sen_p, d_table_p, d_syn0_p, d_syn1neg_p,
              d_random_p, item,
              sm.get_multi_ptr<sycl::access::decorated::no>().get(),
              expTable_p);
        });
  });
  
}
