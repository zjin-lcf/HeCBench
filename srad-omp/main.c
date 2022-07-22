#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "./main.h"
#include "./util/graphics/graphics.h"
#include "./util/graphics/resize.h"
#include "./util/timer/timer.h"

int main(int argc, char* argv []) {

  // time
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;
  long long time7;
  long long time8;
  long long time9;
  long long time10;
  long long time11;
  long long time12;

  time0 = get_time();

  // inputs image, input paramenters
  fp* image_ori;                      // originalinput image
  int image_ori_rows;
  int image_ori_cols;
  long image_ori_elem;

  // inputs image, input paramenters
  fp* image;                          // input image
  int Nr,Nc;                          // IMAGE nbr of rows/cols/elements
  long Ne;

  // algorithm parameters
  int niter;                          // nbr of iterations
  fp lambda;                          // update step size

  // size of IMAGE
  int r1,r2,c1,c2;                    // row/col coordinates of uniform ROI
  long NeROI;                         // ROI nbr of elements

  // surrounding pixel indicies
  int* iN, *iS, *jE, *jW;

  // counters
  int iter;   // primary loop
  long i,j;     // image row/col

  // memory sizes
  int mem_size_i;
  int mem_size_j;

  int blocks_x;
  int blocks_work_size, blocks_work_size2;
  size_t local_work_size;
  int no;
  int mul;
  fp meanROI;
  fp meanROI2;
  fp varROI;
  fp q0sqr;

  time1 = get_time();

  if(argc != 5){
    printf("Usage: %s <repeat> <lambda> <number of rows> <number of columns>\n", argv[0]);
    return 1;
  }
  else{
    niter = atoi(argv[1]);
    lambda = atof(argv[2]);
    Nr = atoi(argv[3]);
    Nc = atoi(argv[4]);
  }

  time2 = get_time();

  //================================================================================80
  //   READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
  //================================================================================80

  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = image_ori_rows * image_ori_cols;
  image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

  const char* input_image_path = "../data/srad/image.pgm";
  if ( !read_graphics( input_image_path, image_ori, image_ori_rows, image_ori_cols, 1) ) {
    printf("ERROR: failed to read input image at %s\n", input_image_path);
    if (image_ori != NULL) free(image_ori);
    return -1; // exit on file i/o error 
  }

  time3 = get_time();

  //================================================================================80
  //   RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
  //================================================================================80

  Ne = Nr*Nc;

  image = (fp*)malloc(sizeof(fp) * Ne);

  resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

  time4 = get_time();

  //   SETUP

  // variables
  r1     = 0;      // top row index of ROI
  r2     = Nr - 1; // bottom row index of ROI
  c1     = 0;      // left column index of ROI
  c2     = Nc - 1; // right column index of ROI

  // ROI image size
  NeROI = (r2-r1+1)*(c2-c1+1);                      // number of elements in ROI, ROI size

  // allocate variables for surrounding pixels
  mem_size_i = sizeof(int) * Nr;                      //
  iN = (int *)malloc(mem_size_i) ;                    // north surrounding element
  iS = (int *)malloc(mem_size_i) ;                    // south surrounding element
  mem_size_j = sizeof(int) * Nc;                      //
  jW = (int *)malloc(mem_size_j) ;                    // west surrounding element
  jE = (int *)malloc(mem_size_j) ;                    // east surrounding element

  // N/S/W/E indices of surrounding pixels (every element of IMAGE)
  for (i=0; i<Nr; i++) {
    iN[i] = i-1;                            // holds index of IMAGE row above
    iS[i] = i+1;                            // holds index of IMAGE row below
  }
  for (j=0; j<Nc; j++) {
    jW[j] = j-1;                            // holds index of IMAGE column on the left
    jE[j] = j+1;                            // holds index of IMAGE column on the right
  }

  // N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
  iN[0]    = 0;                             // changes IMAGE top row index from -1 to 0
  iS[Nr-1] = Nr-1;                          // changes IMAGE bottom row index from Nr to Nr-1 
  jW[0]    = 0;                             // changes IMAGE leftmost column index from -1 to 0
  jE[Nc-1] = Nc-1;                          // changes IMAGE rightmost column index from Nc to Nc-1

  fp *dN = (fp*) malloc (sizeof(fp)*Ne);
  fp *dS = (fp*) malloc (sizeof(fp)*Ne);
  fp *dW = (fp*) malloc (sizeof(fp)*Ne);
  fp *dE = (fp*) malloc (sizeof(fp)*Ne);
  fp *c = (fp*) malloc (sizeof(fp)*Ne);
  fp *sums = (fp*) malloc (sizeof(fp)*Ne);
  fp *sums2 = (fp*) malloc (sizeof(fp)*Ne);

  // threads
  local_work_size = NUMBER_THREADS;

  // workgroups
  blocks_x = Ne/(int)local_work_size;
  if (Ne % (int)local_work_size != 0){ // compensate for division remainder above by adding one grid
    blocks_x = blocks_x + 1;                                  
  }
  blocks_work_size = blocks_x;

  time5 = get_time();

  //================================================================================80
  //   COPY INPUT TO GPU
  //================================================================================80
  #pragma omp target data map(to: image[0:Ne])\
                          map(to: iN[0:Nr], iS[0:Nr], jE[0:Nc], jW[0:Nc])\
                          map(alloc: dN[0:Ne], dS[0:Ne], dW[0:Ne], dE[0:Ne], \
                                     c[0:Ne], sums[0:Ne], sums2[0:Ne])
  {
  time6 = get_time();

  #pragma omp target teams distribute parallel for \
  num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
  for (int ei = 0; ei < Ne; ei++)
    image[ei] = expf(image[ei]/(fp)255); // exponentiate input IMAGE and copy to output image

  time7 = get_time();

  for (iter=0; iter<niter; iter++){ // do for the number of iterations input parameter
    #pragma omp target teams distribute parallel for \
    num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
    for (int ei = 0; ei < Ne; ei++) {
      sums[ei] = image[ei];
      sums2[ei] = image[ei]*image[ei];
    }

    blocks_work_size2 = blocks_work_size;  // original number of blocks
    no = Ne;  // original number of sum elements
    mul = 1;  // original multiplier

    while(blocks_work_size2 != 0){

      // reduce kernel
      #pragma omp target teams num_teams(blocks_work_size2) thread_limit(NUMBER_THREADS)
      {
        fp psum[NUMBER_THREADS];
        fp psum2[NUMBER_THREADS];
        #pragma omp parallel 
        {
          int bx = omp_get_team_num();
          int tx = omp_get_thread_num();
          int ei = (bx*NUMBER_THREADS)+tx;// unique thread id, more threads than actual elements !!!
          int nf = NUMBER_THREADS-(blocks_work_size2*NUMBER_THREADS-no);// number of elements assigned to last block
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
            if(bx != (blocks_work_size2 - 1)){                      //
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
      no = blocks_work_size2;  
      if(blocks_work_size2 == 1){
          blocks_work_size2 = 0;
      }
      else{
        mul = mul * NUMBER_THREADS; // update the increment
        blocks_x = blocks_work_size2/(int)local_work_size; // number of blocks
        if (blocks_work_size2 % (int)local_work_size != 0){ // compensate for division remainder above by adding one grid
            blocks_x = blocks_x + 1;
        }
        blocks_work_size2 = blocks_x;
      }
    } // while

    #pragma omp target update from (sums[0:1])
    #pragma omp target update from (sums2[0:1])

    // calculate statistics

    meanROI  = sums[0] / (fp)(NeROI); // gets mean (average) value of element in ROI
    meanROI2 = meanROI * meanROI;
    varROI = (sums2[0] / (fp)(NeROI)) - meanROI2; // gets variance of ROI                
    q0sqr = varROI / meanROI2; // gets standard deviation of ROI

    // set arguments that were uptaded in this loop
    #pragma omp target teams distribute parallel for \
    num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
    for (int ei = 0; ei < Ne; ei++) {
      // figure out row/col location in new matrix
      int row = (ei+1) % Nr - 1; // (0-n) row
      int col = (ei+1) / Nr + 1 - 1; // (0-n) column
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
      fp d_num  = ((fp)0.5*d_G2) - (((fp)1.0/(fp)16.0)*(d_L*d_L)) ;            // num (based on gradient and laplacian)
      fp d_den  = (fp)1 + ((fp)0.25*d_L);                        // den (based on laplacian)
      fp d_qsqr = d_num/(d_den*d_den);                    // qsqr (based on num and den)

      // diffusion coefficent (equ 33) (every element of IMAGE)
      d_den = (d_qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;        // den (based on qsqr and q0sqr)
      fp d_c_loc = (fp)1.0 / ((fp)1.0+d_den) ;                    // diffusion coefficient (based on den)

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

    #pragma omp target teams distribute parallel for \
    num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
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
      image[ei] += (fp)0.25*lambda*d_D; // updates image (based on input time step and divergence)
    }
  }

  time8 = get_time();

  //   Compress Kernel - SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS

  #pragma omp target teams distribute parallel for \
  num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
  for (int ei = 0; ei < Ne; ei++)
    image[ei] = logf(image[ei])*(fp)255; // exponentiate input IMAGE and copy to output image

  time9 = get_time();

  #pragma omp target update from (image[0:Ne])

  time10 = get_time();

  //   WRITE OUTPUT IMAGE TO FILE

  write_graphics(
      "./image_out.pgm",
      image,
      Nr,
      Nc,
      1,
      255);

  time11 = get_time();

  //   FREE MEMORY
  
  } // #pragma omp target data map

  free(image_ori);
  free(image);
  free(iN); 
  free(iS); 
  free(jW); 
  free(jE);

  time12 = get_time();

  //  DISPLAY TIMING

  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n",
      (float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n",
      (float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n",
      (float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n", 
      (float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n",
      (float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n",
      (float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n", 
      (float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : COMPUTE (%d iterations)\n", 
      (float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time12-time0) * 100, niter);
  printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n", 
      (float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n", 
      (float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n", 
      (float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f %% : FREE MEMORY\n", 
      (float) (time12-time11) / 1000000, (float) (time12-time11) / (float) (time12-time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", (float) (time12-time0) / 1000000);

  return 0;
}
