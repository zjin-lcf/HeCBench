#include <chrono>
#include <iostream>
#include <fstream>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "device.h"

void ker(const float * cormat, float * upper,int n,
         const sycl::nd_item<1> &item)
{
  size_t idx = item.get_global_id(0);
  if (idx < (size_t)n * n) {
    size_t i = idx/n;
    size_t j = idx%n;
    if(i<j)
    {
      size_t t = n * i - i * (i+1) / 2 + j - i - 1;
      //printf("(%lu, %lu) %lu %lu\n", i, j, t, i*n+j);
      upper[t]=cormat[j*n+i];
    }
  }
}

void ker2(const float * cormat, float * upper, int n1, int n,
          const sycl::nd_item<1> &item)
{
  size_t idx = item.get_global_id(0);
  if (idx < (size_t)n1 * n) {
    size_t i = idx/n;
    size_t j = idx%n;
    if(i<j && i<n1)
    {
      size_t t = n * i - i * (i+1) / 2 + j - i - 1;
      upper[t]=cormat[j*n1+i];
    }
  }
}

int CorMat_singlePass(float *upper_tri, float *data, int N, int L) {
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t M1 = (N-1); //computing the  size of correlaion matrix
  M1 *= N;
  M1 /= 2;
  size_t total=N*N;//size of total correlation matrix

  preprocessing(data, N, L);//Preprocessing fMRI data in CPU

  float * dev_data; //Allocating space in GPU for storing fMRI data
  dev_data = (float *)sycl::malloc_device(sizeof(float) * L * N, q);

  q.memcpy(dev_data, data, sizeof(float) * L * N);

  const float alpha = 1.0;
  const float beta = 0.0;

  float* dev_cormat;//allocating space in GPU for whole correlation matrix
  dev_cormat = sycl::malloc_device<float>(total, q);

  //Performing matrix multiplication (fMRI data to its transpose)
  auto start = std::chrono::steady_clock::now();

  sycl::event stat = oneapi::mkl::blas::column_major::gemm(
              q, oneapi::mkl::transpose::trans,
              oneapi::mkl::transpose::nontrans, N, N, L, alpha, dev_data, L,
              dev_data, L, beta, dev_cormat, N);
  stat.wait();
  auto end = std::chrono::steady_clock::now();
  auto gemm_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  float* dev_upper;//Allocating space for extracting upper triangle part
  dev_upper = sycl::malloc_device<float>(M1, q);

  int block_size=THREAD_BLOCK_SIZE;//number of threads
  size_t grid_size=1+((total-1)/block_size);//number of blocks

  sycl::range<1> lws (block_size);
  sycl::range<1> gws (block_size * grid_size);

  start = std::chrono::steady_clock::now();

  // performing kernel for extracting and reordering
  // correlations from upper triangle
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class k>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      ker(dev_cormat, dev_upper, N, item);
    });
  }).wait();

  end = std::chrono::steady_clock::now();
  auto extract_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(upper_tri, dev_upper, sizeof(float) * M1).wait(); // copying upper triangle correlation matrix data back to CPU
  sycl::free(dev_data, q);
  sycl::free(dev_cormat, q);
  sycl::free(dev_upper, q);

  std::cout << "Kernel time (s)\n"
            << "GEMM: " << gemm_time * 1e-9 << ", "
            << "Extract upper triangle: " << extract_time * 1e-9 << "\n";

  return 1;
}

int CorMat_multiPass(float *upper_tri, float *data, int N, int L) {
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int flag=1;

  preprocessing(data,  N, L);//Preprocessing fMRI data

  float * dev_data;//initializing normalized fMRI data in GPU
  dev_data = (float *)sycl::malloc_device(sizeof(float) * L * N, q);

  q.memcpy(dev_data, data, sizeof(float) * L * N);

  const float alpha = 1.0;
  const float beta = 0.0;

  const size_t free_memory = FREE_MEMORY;
  size_t available_memory = free_memory / sizeof(float) - (long)N * L;

  int block=remaining_B(N, available_memory);
  int N_prime=N;

  float* add_uper_cpu=upper_tri;
  size_t M1,temp,temp2=0,temp3=0;
  int so_far=0;
  int pak=0;
  float* dev_cormat;
  float* dev_upper;
  size_t cormat_fullsize;
  size_t gemm_time = 0, extract_time = 0;

  while(flag==1)
  {
    if(block==N_prime)//checking for the last chunk
      flag=0;

    temp = block;
    temp *= (block +1);
    temp /= 2;
    M1=N_prime;
    M1*=block;
    M1-=temp; //M1 is the size of upper triangle part of chunk

    if(pak!=0)
    {
      sycl::free(dev_upper, q);
      sycl::free(dev_cormat, q);
    }
    cormat_fullsize=block;
    cormat_fullsize*=N_prime;

    dev_cormat = sycl::malloc_device<float>(cormat_fullsize, q);
    dev_upper = sycl::malloc_device<float>(M1, q);

    pak++;

    //multiply block x L to L x N_prime = block x N_prime
    auto start = std::chrono::steady_clock::now();
    sycl::event stat = oneapi::mkl::blas::column_major::gemm(
                q, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, block, N_prime, L, alpha,
                dev_data + (so_far * L), L, dev_data + (so_far * L), L, beta,
                dev_cormat, block);
    stat.wait();
    auto end = std::chrono::steady_clock::now();
    gemm_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    temp2=block;
    temp2*=N_prime;

    int block_size=THREAD_BLOCK_SIZE;
    size_t grid_size=1+((temp2-1)/block_size);

    sycl::range<1> lws (block_size);
    sycl::range<1> gws (block_size * grid_size);

    start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k2>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        ker2(dev_cormat, dev_upper, block, N_prime, item);
      });
    }).wait();

    q.wait();
    end = std::chrono::steady_clock::now();
    extract_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    q.memcpy(add_uper_cpu, dev_upper, sizeof(float) * M1).wait();

    temp3+=M1;
    add_uper_cpu=upper_tri+temp3;
    so_far+=block;

    if(N_prime>block)
    {
      N_prime=N_prime-block;
      block=remaining_B(N_prime, available_memory);

      if(N_prime < block)//checking last chunk
        block=N_prime;
    }
  }
  sycl::free(dev_data, q);
  sycl::free(dev_upper, q);
  sycl::free(dev_cormat, q);

  std::cout << "Total Kernel time (s)\n"
            << "GEMM: " << gemm_time * 1e-9 << ", "
            << "Extract upper triangle: " << extract_time * 1e-9 << "\n";

  return 1;
}
