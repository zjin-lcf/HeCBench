/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>
#include <cmath>
#include <sycl/sycl.hpp>

inline int atomicAdd(int& val, const int delta)
{
  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(val);
  return ref.fetch_add(delta);
}

int main(int argc, char** argv)
{
  unsigned int arrayLength = 52428800;
  unsigned int threads=256;
  if(argc == 3) {
      arrayLength=atoi(argv[1]);
      threads=atoi(argv[2]);
  }

  // launch the kernel N iterations 
  int N = 32;

  std::cout << "Array size: " << arrayLength*sizeof(int)/1024.0/1024.0 << " MB"<<std::endl;
  std::cout << "Thread block size: " << threads << std::endl;
  std::cout << "Repeat the kernel execution  " << N << " times" << std::endl;

  const size_t size_bytes = arrayLength * sizeof(int);

  int* array=(int*)malloc(size_bytes);
  int checksum =0;
  for(int i=0;i<arrayLength;i++) {
      array[i]=rand()%2;
      checksum+=array[i];
  }

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Get device properties
  auto device = q.get_device();
  auto deviceName = device.get_info<sycl::info::device::name>();
  std::cout << "Device name: " << deviceName << std::endl;

  int *d_in = sycl::malloc_device<int>(arrayLength, q);
  q.memcpy(d_in, array, size_bytes);

  int *d_out = sycl::malloc_device<int>(1, q);

  int blocks=std::min((arrayLength+threads-1)/threads,2048u);
  size_t global_work_size = blocks * threads;

  // warmup 
  for(int i=0;i<N;i++) {
    q.memset(d_out, 0, sizeof(int));

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class reduction_v0>(sycl::nd_range<1>(
        sycl::range<1>(global_work_size), sycl::range<1>(threads)), [=] (sycl::nd_item<1> item) {
        int sum = 0;
        int idx = item.get_global_id(0);
        for(int i= idx;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)) {
          sum+=d_in[i];
        }
        atomicAdd(d_out[0],sum);
      });
    });
  }
  q.wait();

  // start timing
  t1 = std::chrono::high_resolution_clock::now();
  for(int i=0;i<N;i++) {
    q.memset(d_out, 0, sizeof(int));

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class reduction_v1>(sycl::nd_range<1>(
        sycl::range<1>(global_work_size), sycl::range<1>(threads)), [=] (sycl::nd_item<1> item) {
        int sum = 0;
        int idx = item.get_global_id(0);
        for(int i= idx;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)) {
          sum+=d_in[i];
        }
        atomicAdd(d_out[0],sum);
      });
    });
  }
  q.wait();
  t2 = std::chrono::high_resolution_clock::now();
  double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  float GB=(float)arrayLength*sizeof(int)*N;
  std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

  int sum;
  q.memcpy(&sum, d_out, sizeof(int)).wait();

  if(sum==checksum)
    std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
  else
    std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  for(int i=0;i<N;i++) {
    q.memset(d_out, 0, sizeof(int));

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class reduction_v2>(sycl::nd_range<1>(
        sycl::range<1>(global_work_size/2), sycl::range<1>(threads)), [=] (sycl::nd_item<1> item) {
        int sum = 0;
        int idx = item.get_global_id(0);
        for(int i= idx*2;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*2) {
          sum+=d_in[i] + d_in[i+1];
        }
        atomicAdd(d_out[0],sum);
      });
    });
  }
  q.wait();
  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  GB=(float)arrayLength*sizeof(int)*N;
  std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

  q.memcpy(&sum, d_out, sizeof(int)).wait();

  if(sum==checksum)
    std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
  else
    std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;
  t1 = std::chrono::high_resolution_clock::now();

  for(int i=0;i<N;i++) {
    q.memset(d_out, 0, sizeof(int));

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class reduction_v4>(sycl::nd_range<1>(
        sycl::range<1>(global_work_size/4), sycl::range<1>(threads)), [=] (sycl::nd_item<1> item) {
        int sum = 0;
        int idx = item.get_global_id(0);
        for(int i= idx*4;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*4) {
          sum+=d_in[i] + d_in[i+1] + d_in[i+2] + d_in[i+3];
        }
        atomicAdd(d_out[0],sum);
      });
    });
  }
  q.wait();
  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  GB=(float)arrayLength*sizeof(int)*N;
  std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

  q.memcpy(&sum, d_out, sizeof(int)).wait();

  if(sum==checksum)
    std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
  else
    std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  for(int i=0;i<N;i++) {
    q.memset(d_out, 0, sizeof(int));

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class reduction_v8>(sycl::nd_range<1>(
        sycl::range<1>(global_work_size/8), sycl::range<1>(threads)), [=] (sycl::nd_item<1> item) {
        int sum = 0;
        int idx = item.get_global_id(0);
        for(int i= idx*8;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*8) {
          sum+=d_in[i] + d_in[i+1] + d_in[i+2] + d_in[i+3] +
               d_in[i+4] + d_in[i+5] + d_in[i+6] + d_in[i+7];
        }
        atomicAdd(d_out[0],sum);
      });
    });
  }
  q.wait();
  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  GB=(float)arrayLength*sizeof(int)*N;
  std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

  q.memcpy(&sum, d_out, sizeof(int)).wait();

  if(sum==checksum)
      std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
  else
      std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  for(int i=0;i<N;i++) {
    q.memset(d_out, 0, sizeof(int));

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class reduction_v16>(sycl::nd_range<1>(
        sycl::range<1>(global_work_size/16), sycl::range<1>(threads)), [=] (sycl::nd_item<1> item) {
        int sum = 0;
        int idx = item.get_global_id(0);
        for(int i= idx*16;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*16) {
          sum+=d_in[i] + d_in[i+1] + d_in[i+2] + d_in[i+3] +
               d_in[i+4] + d_in[i+5] + d_in[i+6] + d_in[i+7] +
               d_in[i+8] + d_in[i+9] + d_in[i+10] + d_in[i+11] +
               d_in[i+12] +d_in[i+13] + d_in[i+14] + d_in[i+15] ;
        }
        atomicAdd(d_out[0],sum);
      });
    });
  }
  q.wait();
  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  GB=(float)arrayLength*sizeof(int)*N;
  std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

  q.memcpy(&sum, d_out, sizeof(int)).wait();

  if(sum==checksum)
      std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
  else
      std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

  free(array);
  sycl::free(d_in, q);
  sycl::free(d_out, q);
  return 0;
}
