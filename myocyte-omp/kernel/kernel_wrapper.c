#include <stdio.h>
#include <string.h>
#include "../common.h"
#include "../util/timer/timer.h"
#include "solver.c"
#include "kernel_wrapper.h"



int 
kernel_wrapper(  int xmax,
    int workload,

    fp ***y,
    fp **x,
    fp **params,
    fp *com)
{

  //======================================================================================================================================================150
  //  VARIABLES
  //======================================================================================================================================================150

  long long time0;
  long long time4;
  long long timecopyin;
  long long timekernel;
  long long timecopyout;

  time0 = get_time();



  //======================================================================================================================================================150
  //  EXECUTION
  //======================================================================================================================================================150

  int status;
  int i;

  // workload = 1
  for(i=0; i<workload; i++){

    status = solver(  y[i],
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

  time4 = get_time();

  printf("Device offloading time:\n");
  printf("%.12f s\n", (float) (time4-time0) / 1000000);

  return 0;

}

