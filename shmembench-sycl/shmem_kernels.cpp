/**
 * shmem_kernels.cu: This file is part of the gpumembench suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <chrono> // timing
#include <stdio.h>
#include "common.h"

using namespace std::chrono;

#define TOTAL_ITERATIONS (1024)
#define BLOCK_SIZE 256

// shared memory swap operation
void shmem_swap(float4 *v1, float4 *v2){
  float4 tmp;
  tmp = *v2;
  *v2 = *v1;
  *v1 = tmp;
}

float4 init_val(int i){
  return float4(i, i+11, i+19, i+23);
}

float4 reduce_vector(float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6){
  return float4(v1.x() + v2.x() + v3.x() + v4.x() + v5.x() + v6.x(), 
                v1.y() + v2.y() + v3.y() + v4.y() + v5.y() + v6.y(),
                v1.z() + v2.z() + v3.z() + v4.z() + v5.z() + v6.z(),
                v1.w() + v2.w() + v3.w() + v4.w() + v5.w() + v6.w());
}

void set_vector(float4 *target, int offset, float4 v){
  target[offset].x() = v.x();
  target[offset].y() = v.y();
  target[offset].z() = v.z();
  target[offset].w() = v.w();
}


void benchmark_shmem(float4 *g_data, float4* shm_buffer, nd_item<1> &item){

  int tid = item.get_local_id(0);
  int blk = item.get_local_range(0);
  int gid = item.get_group(0);
  int globaltid = gid * blk + tid;

  set_vector(shm_buffer, tid+0*blk, init_val(tid));
  set_vector(shm_buffer, tid+1*blk, init_val(tid+1));
  set_vector(shm_buffer, tid+2*blk, init_val(tid+3));
  set_vector(shm_buffer, tid+3*blk, init_val(tid+7));
  set_vector(shm_buffer, tid+4*blk, init_val(tid+13));
  set_vector(shm_buffer, tid+5*blk, init_val(tid+17));

  item.barrier(access::fence_space::local_space);

  #pragma unroll 32
  for(int j=0; j<TOTAL_ITERATIONS; j++){
    shmem_swap(shm_buffer+tid+0*blk, shm_buffer+tid+1*blk);
    shmem_swap(shm_buffer+tid+2*blk, shm_buffer+tid+3*blk);
    shmem_swap(shm_buffer+tid+4*blk, shm_buffer+tid+5*blk);

    item.barrier(access::fence_space::local_space);

    shmem_swap(shm_buffer+tid+1*blk, shm_buffer+tid+2*blk);
    shmem_swap(shm_buffer+tid+3*blk, shm_buffer+tid+4*blk);

    item.barrier(access::fence_space::local_space);
  }

  g_data[globaltid] = reduce_vector(shm_buffer[tid+0*blk], 
                                    shm_buffer[tid+1*blk],
                                    shm_buffer[tid+2*blk],
                                    shm_buffer[tid+3*blk],
                                    shm_buffer[tid+4*blk],
                                    shm_buffer[tid+5*blk]);
}

void shmembenchGPU(double *c, const long size, const int n) {
  const int TOTAL_BLOCKS = size/(BLOCK_SIZE);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<double, 1> cd (size);

  range<1> lws (BLOCK_SIZE);
  range<1> gws (TOTAL_BLOCKS/4 * BLOCK_SIZE);
  auto cd_re = cd.reinterpret<float4>(range<1>(size/2));

  auto start = high_resolution_clock::now();
  for (int i = 0; i < n; i++)
    q.submit([&] (handler &cgh) {
      auto g_data = cd_re.get_access<sycl_discard_write>(cgh);
      accessor<float4, 1, sycl_read_write, access::target::local> shm_buffer(BLOCK_SIZE*6, cgh);
      cgh.parallel_for<class kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        benchmark_shmem(g_data.get_pointer(), shm_buffer.get_pointer(), item);
      }); 
    }); 
  q.wait();
  auto end = high_resolution_clock::now();
  auto time_shmem_128b = duration_cast<nanoseconds>(end - start).count() / (double)n;
  printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6);

  // Copy results back to host memory
  q.submit([&] (handler &cgh) {
    auto acc = cd.get_access<sycl_read>(cgh);
    cgh.copy(acc, c);
  }).wait();

  // simple checksum
  double sum = 0;
  for (long i = 0; i < size; i++) sum += c[i];
  if (sum != 21256458760384741137729978368.00)
    printf("checksum failed\n");

  printf("Memory throughput\n");
  const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(float);
  const long long operations_128bit = (6LL+4*5*TOTAL_ITERATIONS+6)*size/4;

  printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n", 
    (double)operations_bytes / time_shmem_128b,
    (double)operations_128bit / time_shmem_128b);
}
