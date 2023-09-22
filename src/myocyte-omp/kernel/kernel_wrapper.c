#include <stdio.h>
#include <string.h>
#include <chrono>
#include "../common.h"
#include "solver.c"
#include "kernel_wrapper.h"

int kernel_wrapper(
    int xmax,
    int workload,
    fp ***y,
    fp **x,
    fp **params,
    fp *com)
{

  //  VARIABLES

  double timecopyin;
  double timekernel;
  double timecopyout;

  auto offload_start = std::chrono::steady_clock::now();

  int status;
  int i;

  // workload = 1
  for(i=0; i<workload; i++){

    status = solver(
        y[i],
        x[i],
        xmax,
        params[i],
        com,
        &timecopyin,
        &timekernel,
        &timecopyout);

    if(status !=0){
      printf("STATUS: %d\n", status);
    }
  }

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();

  printf("Total kernel execution time %f (s)\n\n", timekernel * 1e-9f);
  printf("Total host to device time %f (s)\n\n", timecopyin * 1e-9f);
  printf("Total device to host time %f (s)\n\n", timecopyout * 1e-9f);
  printf("Device offloading time: %f (s)\n", offload_time * 1e-9);

  return 0;
}
