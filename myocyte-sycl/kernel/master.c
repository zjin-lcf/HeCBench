#include "../common.h"
#include "./kernel_fin.c"
#include "kernel_ecc.sycl"
#include "kernel_cam.sycl"
#include <stdio.h>

void master(
    FP timeinst,
    FP *initvalu,
    FP *parameter,
    FP *finavalu,
    FP *com,

    buffer<FP,1>& d_initvalu,
    buffer<FP,1>& d_finavalu,
    buffer<FP,1>& d_params,
    buffer<FP,1>& d_com,

    queue &command_queue,

    double *timecopyin,
    double *timekernel,
    double *timecopyout)
{

  //	VARIABLES

  // counters
  int i;

  // offset pointers
  int initvalu_offset_ecc;																// 46 points
  int initvalu_offset_Dyad;															// 15 points
  int initvalu_offset_SL;																// 15 points
  int initvalu_offset_Cyt;																// 15 poitns

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
          printf("initvalu %d %f\n", i, initvalu[i]);
  for (int i = 0; i < PARAMETERS; i++)
          printf("params %d %f\n", i, parameter[i]);
  printf("\n");
#endif

  // common variables

  //	COPY DATA TO GPU MEMORY
  auto time0 = std::chrono::steady_clock::now();

  command_queue.submit([&](handler& cgh) {
    accessor<FP, 1, access::mode::write, access::target::global_buffer> 
    d_initvalu_acc(d_initvalu, cgh, range<1>(EQUATIONS), id<1>(0));
    cgh.copy(initvalu, d_initvalu_acc);
  });

  command_queue.submit([&](handler& cgh) {
    accessor<FP, 1, access::mode::write, access::target::global_buffer> 
    d_params_acc(d_params, cgh, range<1>(PARAMETERS), id<1>(0));
    cgh.copy(parameter, d_params_acc);
  });

  command_queue.wait();
  auto time1 = std::chrono::steady_clock::now();

  //	GPU: KERNEL

  //====================================================================================================100
  //	KERNEL EXECUTION PARAMETERS
  //====================================================================================================100

  size_t local_work_size = NUMBER_THREADS;
  size_t global_work_size = 2*NUMBER_THREADS;

  // printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)global_work_size[0]/(int)local_work_size[0], (int)local_work_size[0]);
  command_queue.submit([&](handler &cgh) {
    // Getting write only access to the buffer on a device
    auto d_initvalu_acc = d_initvalu.get_access<sycl_read>(cgh);
    auto d_finavalu_acc = d_finavalu.get_access<sycl_write>(cgh);
    auto d_params_acc = d_params.get_access<sycl_read>(cgh);
    auto d_com_acc = d_com.get_access<sycl_write>(cgh);
    // Executing kernel
    cgh.parallel_for<class sycl_kernel >(
      nd_range<1>(range<1>(global_work_size),
        range<1>(local_work_size)), [=] (nd_item<1> item) {
          #include "kernel.sycl"
    });
  });

  command_queue.wait();
  auto time2 = std::chrono::steady_clock::now();

  //	COPY DATA TO SYSTEM MEMORY

  command_queue.submit([&](handler& cgh) {
    accessor<FP, 1, sycl_read, access::target::global_buffer> 
    d_finavalu_acc(d_finavalu, cgh, range<1>(EQUATIONS), id<1>(0));
    cgh.copy(d_finavalu_acc, finavalu);
  });

  command_queue.submit([&](handler& cgh) {
    accessor<FP, 1, sycl_read, access::target::global_buffer> 
    d_com_acc(d_com, cgh, range<1>(3), id<1>(0));
    cgh.copy(d_com_acc, com);
  });

  command_queue.wait();
  auto time3 = std::chrono::steady_clock::now();

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
          printf("finavalu %d %f\n", i, finavalu[i]);
  for (int i = 0; i < 3; i++)
          printf("%f ", com[i]);
  printf("\n");

#endif

  // accumulate host-to-device, kernel, and device-to-host time
  *timecopyin += std::chrono::duration_cast<std::chrono::nanoseconds>(time1-time0).count();
  *timekernel += std::chrono::duration_cast<std::chrono::nanoseconds>(time2-time1).count();
  *timecopyout += std::chrono::duration_cast<std::chrono::nanoseconds>(time3-time2).count();

  //	CPU: FINAL KERNEL

  initvalu_offset_ecc = 0;
  initvalu_offset_Dyad = 46;
  initvalu_offset_SL = 61;
  initvalu_offset_Cyt = 76;

  kernel_fin(
      initvalu,
      initvalu_offset_ecc,
      initvalu_offset_Dyad,
      initvalu_offset_SL,
      initvalu_offset_Cyt,
      parameter,
      finavalu,
      com[0],
      com[1],
      com[2]);

  //	COMPENSATION FOR NANs and INFs

  for(i=0; i<EQUATIONS; i++){
    if (std::isnan(finavalu[i])){ 
      finavalu[i] = 0.0001; // for NAN set rate of change to 0.0001
    }
    else if (std::isinf(finavalu[i])){ 
      finavalu[i] = 0.0001; // for INF set rate of change to 0.0001
    }
  }
}
