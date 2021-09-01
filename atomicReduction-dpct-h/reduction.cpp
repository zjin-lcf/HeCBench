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

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>
#include <cmath>

void check_cuda_error(void) try {
/*
DPCT1010:2: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
int err = 0;
/*
DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
*/
if (err != 0)
{
/*
DPCT1001:0: The statement could not be removed.
*/
std::cerr << "Error: "
          /*
          DPCT1009:3: SYCL uses exceptions to report errors and does not use the
          error codes. The original code was commented out and a warning string
          was inserted. You need to rewrite this code.
          */
          << "cudaGetErrorString not supported" /*cudaGetErrorString(err)*/
          << std::endl;
        exit(err);
}
}
catch (sycl::exception const &exc) {
std::cerr << exc.what() << "Exception caught at file:" << __FILE__
          << ", line:" << __LINE__ << std::endl;
std::exit(1);
}

void atomic_reduction(int *in, int* out, int arrayLength,
                      sycl::nd_item<3> item_ct1) {
    int sum=int(0);
int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
for (int i = idx; i < arrayLength;
     i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
        sum+=in[i];
    }
sycl::atomic<int>(sycl::global_ptr<int>(out)).fetch_add(sum);
}

void atomic_reduction_v2(int *in, int* out, int arrayLength,
                         sycl::nd_item<3> item_ct1) {
    int sum=int(0);
int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
for (int i = idx * 2; i < arrayLength;
     i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2) * 2) {
        sum+=in[i] + in[i+1];
    }
sycl::atomic<int>(sycl::global_ptr<int>(out)).fetch_add(sum);
}

void atomic_reduction_v4(int *in, int* out, int arrayLength,
                         sycl::nd_item<3> item_ct1) {
    int sum=int(0);
int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
for (int i = idx * 4; i < arrayLength;
     i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2) * 4) {
        sum+=in[i] + in[i+1] + in[i+2] + in[i+3];
    }
sycl::atomic<int>(sycl::global_ptr<int>(out)).fetch_add(sum);
}
void atomic_reduction_v8(int *in, int* out, int arrayLength,
                         sycl::nd_item<3> item_ct1) {
    int sum=int(0);
int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
for (int i = idx * 8; i < arrayLength;
     i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2) * 8) {
        sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7];
    }
sycl::atomic<int>(sycl::global_ptr<int>(out)).fetch_add(sum);
}

void atomic_reduction_v16(int *in, int* out, int arrayLength,
                          sycl::nd_item<3> item_ct1) {
    int sum=int(0);
int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
for (int i = idx * 16; i < arrayLength; i += item_ct1.get_local_range().get(2) *
                                             item_ct1.get_group_range(2) * 16) {
        sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7] 
            +in[i+8] +in[i+9] +in[i+10] +in[i+11] +in[i+12] +in[i+13] +in[i+14] +in[i+15] ;
    }
sycl::atomic<int>(sycl::global_ptr<int>(out)).fetch_add(sum);
}

int main(int argc, char** argv)
{
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
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
    int *in, *out;

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;


    long long size=sizeof(int)*arrayLength;

    // Get device properties
dpct::device_info props;
dpct::dev_mgr::instance().get_device(0).get_device_info(props);
std::cout << "Device name: " << props.get_name() << std::endl;

dpct::dpct_malloc(&in, size);
dpct::dpct_malloc(&out, sizeof(int));
    check_cuda_error();

dpct::dpct_memcpy(in, array, arrayLength * sizeof(int), dpct::host_to_device);
dev_ct1.queues_wait_and_throw();
    check_cuda_error();

int blocks = std::min((arrayLength + threads - 1) / threads, 2048u);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
dpct::async_dpct_memset(out, 0, sizeof(int));
{
dpct::buffer_t in_buf_ct0 = dpct::get_buffer(in);
dpct::buffer_t out_buf_ct1 = dpct::get_buffer(out);
q_ct1.submit([&](sycl::handler &cgh) {
auto in_acc_ct0 = in_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
auto out_acc_ct1 = out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                       sycl::range<3>(1, 1, threads),
                                   sycl::range<3>(1, 1, threads)),
                 [=](sycl::nd_item<3> item_ct1) {
                 atomic_reduction((int *)(&in_acc_ct0[0]),
                                  (int *)(&out_acc_ct1[0]), arrayLength,
                                  item_ct1);
                 });
});
}
        check_cuda_error();
dev_ct1.queues_wait_and_throw();
        check_cuda_error();
    }
    t2 = std::chrono::high_resolution_clock::now();
    double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    float GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    int sum;
dpct::dpct_memcpy(&sum, out, sizeof(int), dpct::device_to_host);
    check_cuda_error();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
dpct::async_dpct_memset(out, 0, sizeof(int));
{
dpct::buffer_t in_buf_ct0 = dpct::get_buffer(in);
dpct::buffer_t out_buf_ct1 = dpct::get_buffer(out);
q_ct1.submit([&](sycl::handler &cgh) {
auto in_acc_ct0 = in_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
auto out_acc_ct1 = out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                       sycl::range<3>(1, 1, threads),
                                   sycl::range<3>(1, 1, threads)),
                 [=](sycl::nd_item<3> item_ct1) {
                 atomic_reduction_v2((int *)(&in_acc_ct0[0]),
                                     (int *)(&out_acc_ct1[0]), arrayLength,
                                     item_ct1);
                 });
});
}
        check_cuda_error();
dev_ct1.queues_wait_and_throw();
        check_cuda_error();
    }
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

dpct::dpct_memcpy(&sum, out, sizeof(int), dpct::device_to_host);
    check_cuda_error();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();

    for(int i=0;i<N;i++) {
dpct::async_dpct_memset(out, 0, sizeof(int));
{
dpct::buffer_t in_buf_ct0 = dpct::get_buffer(in);
dpct::buffer_t out_buf_ct1 = dpct::get_buffer(out);
q_ct1.submit([&](sycl::handler &cgh) {
auto in_acc_ct0 = in_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
auto out_acc_ct1 = out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                       sycl::range<3>(1, 1, threads),
                                   sycl::range<3>(1, 1, threads)),
                 [=](sycl::nd_item<3> item_ct1) {
                 atomic_reduction_v4((int *)(&in_acc_ct0[0]),
                                     (int *)(&out_acc_ct1[0]), arrayLength,
                                     item_ct1);
                 });
});
}
        check_cuda_error();
dev_ct1.queues_wait_and_throw();
        check_cuda_error();
    }
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

dpct::dpct_memcpy(&sum, out, sizeof(int), dpct::device_to_host);
    check_cuda_error();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
dpct::async_dpct_memset(out, 0, sizeof(int));
{
dpct::buffer_t in_buf_ct0 = dpct::get_buffer(in);
dpct::buffer_t out_buf_ct1 = dpct::get_buffer(out);
q_ct1.submit([&](sycl::handler &cgh) {
auto in_acc_ct0 = in_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
auto out_acc_ct1 = out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                       sycl::range<3>(1, 1, threads),
                                   sycl::range<3>(1, 1, threads)),
                 [=](sycl::nd_item<3> item_ct1) {
                 atomic_reduction_v8((int *)(&in_acc_ct0[0]),
                                     (int *)(&out_acc_ct1[0]), arrayLength,
                                     item_ct1);
                 });
});
}
        check_cuda_error();
dev_ct1.queues_wait_and_throw();
        check_cuda_error();
    }
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

dpct::dpct_memcpy(&sum, out, sizeof(int), dpct::device_to_host);
    check_cuda_error();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
dpct::async_dpct_memset(out, 0, sizeof(int));
{
dpct::buffer_t in_buf_ct0 = dpct::get_buffer(in);
dpct::buffer_t out_buf_ct1 = dpct::get_buffer(out);
q_ct1.submit([&](sycl::handler &cgh) {
auto in_acc_ct0 = in_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
auto out_acc_ct1 = out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                       sycl::range<3>(1, 1, threads),
                                   sycl::range<3>(1, 1, threads)),
                 [=](sycl::nd_item<3> item_ct1) {
                 atomic_reduction_v16((int *)(&in_acc_ct0[0]),
                                      (int *)(&out_acc_ct1[0]), arrayLength,
                                      item_ct1);
                 });
});
}
        check_cuda_error();
dev_ct1.queues_wait_and_throw();
        check_cuda_error();
    }
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

dpct::dpct_memcpy(&sum, out, sizeof(int), dpct::device_to_host);
    check_cuda_error();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

dpct::dpct_free(in);
dpct::dpct_free(out);
    check_cuda_error();

    free(array);

}
