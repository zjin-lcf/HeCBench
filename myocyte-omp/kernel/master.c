#include <stdio.h>
#include <math.h>
#include "../common.h"
#include "kernel_fin.c"
#include "kernel_ecc.h"
#include "kernel_cam.h"

void 
master(  fp timeinst,
    fp *initvalu,
    fp *params,
    fp *finavalu,
    fp *com,
    long long *timecopyin,
    long long *timekernel,
    long long *timecopyout)
{

  //======================================================================================================================================================150
  //  VARIABLES
  //======================================================================================================================================================150

  //timer
  long long time0;
  long long time1;
  long long time2;
  long long time3;

  // counters
  int i;

  // offset pointers
  int initvalu_offset_ecc;                                // 46 points
  int initvalu_offset_Dyad;                              // 15 points
  int initvalu_offset_SL;                                // 15 points
  int initvalu_offset_Cyt;                                // 15 poitns

  // common variables
  time0 = get_time();

  //======================================================================================================================================================150
  //  COPY DATA TO GPU MEMORY
  //======================================================================================================================================================150

  //====================================================================================================100
  //  initvalu
  //====================================================================================================100

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
	  printf("initvalu %d %f\n", i, initvalu[i]);
  for (int i = 0; i < PARAMETERS; i++)
	  printf("params %d %f\n", i, params[i]);
  printf("\n");
#endif

#pragma omp target update to (initvalu[0:EQUATIONS]) 
#pragma omp target update to (params[0:PARAMETERS]) 


  time1 = get_time();

  //  GPU: KERNEL

#pragma omp target teams num_teams(2) thread_limit(NUMBER_THREADS)
  {
#pragma omp parallel
    {
      int bx = omp_get_team_num();
      int tx = omp_get_thread_num();

      // pointers
      int valu_offset;                                  // inivalu and finavalu offset
      int params_offset;                                  // parameters offset
      int com_offset;                                    // kernel1-kernel2 communication offset

      // module parameters
      fp CaDyad;                                      // from ECC model, *** Converting from [mM] to [uM] ***
      fp CaSL;                                      // from ECC model, *** Converting from [mM] to [uM] ***
      fp CaCyt;                                      // from ECC model, *** Converting from [mM] to [uM] ***

      //====================================================================================================100
      //  ECC
      //====================================================================================================100

      // limit to useful threads
      if(bx == 0){                                    // first processor runs ECC

        if(tx == 0){                                  // only 1 thread runs it, since its a sequential code

          // thread offset
          valu_offset = 0;                              //
          // ecc function
          kernel_ecc(  timeinst,
              initvalu,
              finavalu,
              valu_offset,
              params);

        }

      }

      //====================================================================================================100
      //  CAM x 3
      //====================================================================================================100

      // limit to useful threads
      else if(bx == 1){                                  // second processor runs CAMs (in parallel with ECC)

        if(tx == 0){                                  // only 1 thread runs it, since its a sequential code

          // specific
          valu_offset = 46;
          params_offset = 0;
          com_offset = 0;
          CaDyad = initvalu[35]*1e3;                        // from ECC model, *** Converting from [mM] to [uM] ***
          // cam function for Dyad
          kernel_cam(  timeinst,
              initvalu,
              finavalu,
              valu_offset,
              params,
              params_offset,
              com,
              com_offset,
              CaDyad);

          // specific
          valu_offset = 61;
          params_offset = 5;
          com_offset = 1;
          CaSL = initvalu[36]*1e3;                          // from ECC model, *** Converting from [mM] to [uM] ***
          // cam function for Dyad
          kernel_cam(  timeinst,
              initvalu,
              finavalu,
              valu_offset,
              params,
              params_offset,
              com,
              com_offset,
              CaSL);

          // specific
          valu_offset = 76;
          params_offset = 10;
          com_offset = 2;
          CaCyt = initvalu[37]*1e3;                    // from ECC model, *** Converting from [mM] to [uM] ***
          // cam function for Dyad
          kernel_cam(  timeinst,
              initvalu,
              finavalu,
              valu_offset,
              params,
              params_offset,
              com,
              com_offset,
              CaCyt);
        }
      }
    }
  }
  time2 = get_time();


#pragma omp target update from (finavalu[0:EQUATIONS]) 
#pragma omp target update from (com[0:3]) 

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
	  printf("finavalu %d %f\n", i, finavalu[i]);
  for (int i = 0; i < 3; i++)
	  printf("%f ", com[i]);
  printf("\n");

#endif


  time3 = get_time();

  //======================================================================================================================================================150
  //  CPU: FINAL KERNEL
  //======================================================================================================================================================150

  // *copyin_time,
  // *kernel_time,
  // *copyout_time)

  timecopyin[0] = timecopyin[0] + (time1-time0);
  timekernel[0] = timekernel[0] + (time2-time1);
  timecopyout[0] = timecopyout[0] + (time3-time2);

  //======================================================================================================================================================150
  //  CPU: FINAL KERNEL
  //======================================================================================================================================================150

  initvalu_offset_ecc = 0;                        // 46 points
  initvalu_offset_Dyad = 46;                        // 15 points
  initvalu_offset_SL = 61;                        // 15 points
  initvalu_offset_Cyt = 76;                        // 15 poitns

  kernel_fin(  initvalu,
      initvalu_offset_ecc,
      initvalu_offset_Dyad,
      initvalu_offset_SL,
      initvalu_offset_Cyt,
      params,
      finavalu,
      com[0],
      com[1],
      com[2]);

  //======================================================================================================================================================150
  //  COMPENSATION FOR NANs and INFs
  //======================================================================================================================================================150

  for(i=0; i<EQUATIONS; i++){
    if (isnan(finavalu[i])){ 
      finavalu[i] = 0.0001;                        // for NAN set rate of change to 0.0001
    }
    else if (isinf(finavalu[i])){ 
      finavalu[i] = 0.0001;                        // for INF set rate of change to 0.0001
    }
  }
}

