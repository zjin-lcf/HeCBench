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
#include "common.h"

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

    int* array=(int*)malloc(arrayLength*sizeof(int));
    int checksum =0;
    for(int i=0;i<arrayLength;i++) {
        array[i]=rand()%2;
        checksum+=array[i];
    }

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;

    {
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();

    // Get device properties
    auto device = q.get_device();
    auto deviceName = device.get_info<info::device::name>();
    std::cout << "Device name: " << deviceName << std::endl;

    buffer<int, 1> d_in(array, arrayLength, props);
    buffer<int, 1> d_out(1);

    int blocks=std::min((arrayLength+threads-1)/threads,2048u);
    size_t global_work_size = blocks * threads;

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
      q.submit([&](handler& cgh) { 
        auto out = d_out.get_access<sycl_write>(cgh);
        cgh.fill(out, 0);
      });

      q.submit([&](handler& cgh) { 
        auto in = d_in.get_access<sycl_read>(cgh);
        auto out = d_out.get_access<sycl_atomic>(cgh);
        cgh.parallel_for<class reduction>(nd_range<1>(
          range<1>(global_work_size), range<1>(threads)), [=] (nd_item<1> item) {
          int sum = 0;
          int idx = item.get_global_id(0);
          for(int i= idx;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)) {
            sum+=in[i];
          }
          atomic_fetch_add(out[0],sum);
          });
      });
    }
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    float GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    int sum;
    q.submit([&](handler& cgh) { 
      auto out = d_out.get_access<sycl_read>(cgh);
      cgh.copy(out, &sum);
    });
    q.wait();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
      q.submit([&](handler& cgh) { 
        auto out = d_out.get_access<sycl_write>(cgh);
        cgh.fill(out, 0);
      });

      q.submit([&](handler& cgh) { 
        auto in = d_in.get_access<sycl_read>(cgh);
        auto out = d_out.get_access<sycl_atomic>(cgh);
        cgh.parallel_for<class reduction>(nd_range<1>(
          range<1>(global_work_size), range<1>(threads)), [=] (nd_item<1> item) {
          int sum = 0;
          int idx = item.get_global_id(0);
          for(int i= idx*2;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*2) {
            sum+=in[i] + in[i+1];
          }
          atomic_fetch_add(out[0],sum);
          });
      });
    }
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    q.submit([&](handler& cgh) { 
      auto out = d_out.get_access<sycl_read>(cgh);
      cgh.copy(out, &sum);
    });
    q.wait();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();

    for(int i=0;i<N;i++) {
      q.submit([&](handler& cgh) { 
        auto out = d_out.get_access<sycl_write>(cgh);
        cgh.fill(out, 0);
      });

      q.submit([&](handler& cgh) { 
        auto in = d_in.get_access<sycl_read>(cgh);
        auto out = d_out.get_access<sycl_atomic>(cgh);
        cgh.parallel_for<class reduction>(nd_range<1>(
          range<1>(global_work_size), range<1>(threads)), [=] (nd_item<1> item) {
          int sum = 0;
          int idx = item.get_global_id(0);
          for(int i= idx*4;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*4) {
            sum+=in[i] + in[i+1] + in[i+2] + in[i+3];
          }
          atomic_fetch_add(out[0],sum);
          });
      });
    }
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    q.submit([&](handler& cgh) { 
      auto out = d_out.get_access<sycl_read>(cgh);
      cgh.copy(out, &sum);
    });
    q.wait();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
      q.submit([&](handler& cgh) { 
        auto out = d_out.get_access<sycl_write>(cgh);
        cgh.fill(out, 0);
      });

      q.submit([&](handler& cgh) { 
        auto in = d_in.get_access<sycl_read>(cgh);
        auto out = d_out.get_access<sycl_atomic>(cgh);
        cgh.parallel_for<class reduction>(nd_range<1>(
          range<1>(global_work_size), range<1>(threads)), [=] (nd_item<1> item) {
          int sum = 0;
          int idx = item.get_global_id(0);
          for(int i= idx*8;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*8) {
            sum+=in[i] + in[i+1] + in[i+2] + in[i+3] + in[i+4] + in[i+5] + in[i+6] + in[i+7];
          }
          atomic_fetch_add(out[0],sum);
          });
      });
   }
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    q.submit([&](handler& cgh) { 
      auto out = d_out.get_access<sycl_read>(cgh);
      cgh.copy(out, &sum);
    });
    q.wait();


    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
      q.submit([&](handler& cgh) { 
        auto out = d_out.get_access<sycl_write>(cgh);
        cgh.fill(out, 0);
      });

      q.submit([&](handler& cgh) { 
        auto in = d_in.get_access<sycl_read>(cgh);
        auto out = d_out.get_access<sycl_atomic>(cgh);
        cgh.parallel_for<class reduction>(nd_range<1>(
          range<1>(global_work_size), range<1>(threads)), [=] (nd_item<1> item) {
          int sum = 0;
          int idx = item.get_global_id(0);
          for(int i= idx*16;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*16) {
            sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7] +
	         in[i+8] +in[i+9] +in[i+10] +in[i+11] +in[i+12] +in[i+13] +in[i+14] +in[i+15] ;
          }
          atomic_fetch_add(out[0],sum);
          });
      });
    }
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    q.submit([&](handler& cgh) { 
      auto out = d_out.get_access<sycl_read>(cgh);
      cgh.copy(out, &sum);
    });
    q.wait();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    }

    free(array);

}
