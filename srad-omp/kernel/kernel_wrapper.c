#include <stdio.h>
#include <string.h>
#include <iostream>
#include <omp.h>
#include <math.h>
#include "./../main.h"


  void 
kernel_wrapper(  fp* image,                      // input image
    int Nr,                        // IMAGE nbr of rows
    int Nc,                        // IMAGE nbr of cols
    long Ne,                      // IMAGE nbr of elem
    int niter,                      // nbr of iterations
    fp lambda,                      // update step size
    long NeROI,                      // ROI nbr of elements
    int* iN,
    int* iS,
    int* jE,
    int* jW,
    int iter)                      // primary loop

{
 fp *dN = (fp*) malloc (sizeof(fp)*Ne);
 fp *dS = (fp*) malloc (sizeof(fp)*Ne);
 fp *dW = (fp*) malloc (sizeof(fp)*Ne);
 fp *dE = (fp*) malloc (sizeof(fp)*Ne);
 fp *c = (fp*) malloc (sizeof(fp)*Ne);
 fp *sums = (fp*) malloc (sizeof(fp)*Ne);
 fp *sums2 = (fp*) malloc (sizeof(fp)*Ne);

#pragma omp target data map(tofrom: image[0:Ne])\
  map(to: iN[0:Nr], iS[0:Nr], jE[0:Nc], jW[0:Nc])\
  map(alloc: dN[0:Ne], dS[0:Ne], dW[0:Ne], dE[0:Ne], \
      c[0:Ne], sums[0:Ne], sums2[0:Ne])
  {

    //======================================================================================================================================================150
    //   KERNEL EXECUTION PARAMETERS
    //======================================================================================================================================================150

    // threads
    size_t threads;
    threads = NUMBER_THREADS;

    // workgroups
    int blocks_work_size;
    int blocks_x = Ne/(int)threads;
    if (Ne % (int)threads != 0){                        // compensate for division remainder above by adding one grid
      blocks_x = blocks_x + 1;                                  
    }
    blocks_work_size = blocks_x;

    printf("max # of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n",
        (int)(blocks_work_size), (int)threads);

    //======================================================================================================================================================150
    //   Extract Kernel - SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
    //======================================================================================================================================================150

#ifdef DEBUG
    for (long i = 0; i < 16; i++)
      printf("before extract: %f\n",image[i]);
    printf("\n");
#endif

#pragma omp target teams distribute parallel for num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
    for (int ei = 0; ei < Ne; ei++)
      image[ei] = expf(image[ei]/255); // exponentiate input IMAGE and copy to output image

    int blocks2_work_size;
    long no;
    int mul;
    fp meanROI;
    fp meanROI2;
    fp varROI;
    fp q0sqr;


    printf("Iterations Progress: ");


    // execute main loop
    for (iter=0; iter<niter; iter++){ // do for the number of iterations input parameter

      printf("%d ", iter);
      fflush(NULL);

      // Prepare kernel
#pragma omp target teams distribute parallel for num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
      for (int ei = 0; ei < Ne; ei++) {
        sums[ei] = image[ei];
        sums2[ei] = image[ei]*image[ei];
      }

      // initial values
      blocks2_work_size = blocks_work_size;              // original number of blocks
      no = Ne;                            // original number of sum elements
      mul = 1;                            // original multiplier

      // loop
      while(blocks2_work_size != 0){
#ifdef DEBUG
        printf("max # of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n",
            blocks2_work_size, (int)threads);
#endif
#pragma omp target teams num_teams(blocks2_work_size) thread_limit(NUMBER_THREADS)
        {
          fp psum[NUMBER_THREADS];
          fp psum2[NUMBER_THREADS];
#pragma omp parallel 
          {

            int bx = omp_get_team_num();
            int tx = omp_get_thread_num();
            int ei = (bx*NUMBER_THREADS)+tx;// unique thread id, more threads than actual elements !!!
            int nf = NUMBER_THREADS-(blocks2_work_size*NUMBER_THREADS-no);// number of elements assigned to last block
            int df = 0;// divisibility factor for the last block

            // counters
            int i;

            // copy data to shared memory
            if(ei<no){// do only for the number of elements, omit extra threads

              psum[tx] = sums[ei*mul];
              psum2[tx] = sums2[ei*mul];

            }

#pragma omp barrier

            // reduction of sums if all blocks are full (rare case)  
            if(nf == NUMBER_THREADS){
              // sum of every 2, 4, ..., NUMBER_THREADS elements
              for(i=2; i<=NUMBER_THREADS; i=2*i){
                // sum of elements
                if((tx+1) % i == 0){                      // every ith
                  psum[tx] = psum[tx] + psum[tx-i/2];
                  psum2[tx] = psum2[tx] + psum2[tx-i/2];
                }
                // synchronization
#pragma omp barrier
              }
              // final sumation by last thread in every block
              if(tx==(NUMBER_THREADS-1)){                      // block result stored in global memory
                sums[bx*mul*NUMBER_THREADS] = psum[tx];
                sums2[bx*mul*NUMBER_THREADS] = psum2[tx];
              }
            }
            // reduction of sums if last block is not full (common case)
            else{ 
              // for full blocks (all except for last block)
              if(bx != (blocks2_work_size - 1)){                      //
                // sum of every 2, 4, ..., NUMBER_THREADS elements
                for(i=2; i<=NUMBER_THREADS; i=2*i){                //
                  // sum of elements
                  if((tx+1) % i == 0){                    // every ith
                    psum[tx] = psum[tx] + psum[tx-i/2];
                    psum2[tx] = psum2[tx] + psum2[tx-i/2];
                  }
                  // synchronization
#pragma omp barrier
                }
                // final sumation by last thread in every block
                if(tx==(NUMBER_THREADS-1)){                    // block result stored in global memory
                  sums[bx*mul*NUMBER_THREADS] = psum[tx];
                  sums2[bx*mul*NUMBER_THREADS] = psum2[tx];
                }
              }
              // for not full block (last block)
              else{                                //
                // figure out divisibility
                for(i=2; i<=NUMBER_THREADS; i=2*i){                //
                  if(nf >= i){
                    df = i;
                  }
                }
                // sum of every 2, 4, ..., NUMBER_THREADS elements
                for(i=2; i<=df; i=2*i){                      //
                  // sum of elements (only busy threads)
                  if((tx+1) % i == 0 && tx<df){                // every ith
                    psum[tx] = psum[tx] + psum[tx-i/2];
                    psum2[tx] = psum2[tx] + psum2[tx-i/2];
                  }
                  // synchronization (all threads)
#pragma omp barrier
                }
                // remainder / final summation by last thread
                if(tx==(df-1)){                    //
                  // compute the remainder and final summation by last busy thread
                  for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf; i++){            //
                    psum[tx] = psum[tx] + sums[i];
                    psum2[tx] = psum2[tx] + sums2[i];
                  }
                  // final sumation by last thread in every block
                  sums[bx*mul*NUMBER_THREADS] = psum[tx];
                  sums2[bx*mul*NUMBER_THREADS] = psum2[tx];
                }
              }
            }
          }
        }

        // update execution parameters
        no = blocks2_work_size;  
        if(blocks2_work_size == 1){
          blocks2_work_size = 0;
        }
        else{
          mul = mul * NUMBER_THREADS;                    // update the increment
          blocks_x = blocks2_work_size/(int)threads;      // number of blocks
          if (blocks2_work_size % (int)threads != 0){      // compensate for division remainder above by adding one grid
            blocks_x = blocks_x + 1;
          }
          blocks2_work_size = blocks_x;
        }
      } // while

#pragma omp target update from (sums[0:1])
#pragma omp target update from (sums2[0:1])

#ifdef DEBUG
      printf("total: %f total2: %f\n", sums[0], sums2[0]);
#endif

      //====================================================================================================100
      // calculate statistics
      //====================================================================================================100

      meanROI  = sums[0] / (fp)(NeROI);                    // gets mean (average) value of element in ROI
      meanROI2 = meanROI * meanROI;                    //
      varROI = (sums2[0] / (fp)(NeROI)) - meanROI2;              // gets variance of ROI                
      q0sqr = varROI / meanROI2;                      // gets standard deviation of ROI

      //====================================================================================================100
      // execute srad kernel
      //====================================================================================================100

#pragma omp target teams distribute parallel for num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
      for (int ei = 0; ei < Ne; ei++) {

        // figure out row/col location in new matrix
        int row = (ei+1) % Nr - 1;                          // (0-n) row
        int col = (ei+1) / Nr + 1 - 1;                        // (0-n) column
        if((ei+1) % Nr == 0){
          row = Nr - 1;
          col = col - 1;
        }


        // directional derivatives, ICOV, diffusion coefficent
        fp d_Jc = image[ei];                            // get value of the current element

        // directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
        fp N_loc = image[iN[row] + Nr*col] - d_Jc;            // north direction derivative
        fp S_loc = image[iS[row] + Nr*col] - d_Jc;            // south direction derivative
        fp W_loc = image[row + Nr*jW[col]] - d_Jc;            // west direction derivative
        fp E_loc = image[row + Nr*jE[col]] - d_Jc;            // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        fp d_G2 = (N_loc*N_loc + S_loc*S_loc + W_loc*W_loc + E_loc*E_loc) / (d_Jc*d_Jc);  // gradient (based on derivatives)

        // normalized discrete laplacian (equ 54)
        fp d_L = (N_loc + S_loc + W_loc + E_loc) / d_Jc;      // laplacian (based on derivatives)

        // ICOV (equ 31/35)
        fp d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;            // num (based on gradient and laplacian)
        fp d_den  = 1 + (0.25*d_L);                        // den (based on laplacian)
        fp d_qsqr = d_num/(d_den*d_den);                    // qsqr (based on num and den)

        // diffusion coefficent (equ 33) (every element of IMAGE)
        d_den = (d_qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;        // den (based on qsqr and q0sqr)
        fp d_c_loc = 1.0 / (1.0+d_den) ;                    // diffusion coefficient (based on den)

        // saturate diffusion coefficent to 0-1 range
        if (d_c_loc < 0){                          // if diffusion coefficient < 0
          d_c_loc = 0;                          // ... set to 0
        }
        else if (d_c_loc > 1){                        // if diffusion coefficient > 1
          d_c_loc = 1;                          // ... set to 1
        }

        // save data to global memory
        dN[ei] = N_loc; 
        dS[ei] = S_loc; 
        dW[ei] = W_loc; 
        dE[ei] = E_loc;
        c[ei] = d_c_loc;

      }


      //====================================================================================================100
      // execute srad2 kernel
      //====================================================================================================100

#pragma omp target teams distribute parallel for num_teams(blocks_work_size ) thread_limit(NUMBER_THREADS)
      for (int ei = 0; ei < Ne; ei++){              // make sure that only threads matching jobs run
        // figure out row/col location in new matrix
        int row = (ei+1) % Nr - 1;  // (0-n) row
        int col = (ei+1) / Nr ;     // (0-n) column
        if((ei+1) % Nr == 0){
          row = Nr - 1;
          col = col - 1;
        }

        // diffusion coefficent
        fp d_cN = c[ei];  // north diffusion coefficient
        fp d_cS = c[iS[row] + Nr*col];  // south diffusion coefficient
        fp d_cW = c[ei];  // west diffusion coefficient
        fp d_cE = c[row + Nr * jE[col]];  // east diffusion coefficient

        // divergence (equ 58)
        fp d_D = d_cN*dN[ei] + d_cS*dS[ei] + d_cW*dW[ei] + d_cE*dE[ei];

        // image update (equ 61) (every element of IMAGE)
        image[ei] += 0.25*lambda*d_D; // updates image (based on input time step and divergence)

      }

    }  // for

    // print a newline after the display of iteration numbers
    printf("\n");


    //======================================================================================================================================================150
    //   Compress Kernel - SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
    //======================================================================================================================================================150

#pragma omp target teams distribute parallel for num_teams(blocks_work_size ) thread_limit(NUMBER_THREADS)
    for (int ei = 0; ei < Ne; ei++)
      image[ei] = logf(image[ei])*255; // exponentiate input IMAGE and copy to output image
  }
    //
#ifdef DEBUG
    for (long i = 0; i < 16; i++)
      printf("%f ", image[i]);
    printf("\n");
#endif
 free(dN);
 free(dS);
 free(dW);
 free(dE);
 free(c);
 free(sums);
 free(sums2);
}

