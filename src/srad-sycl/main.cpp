#include <stdio.h>
#include <stdlib.h>
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
  FP* image_ori;                      // originalinput image
  int image_ori_rows;
  int image_ori_cols;
  long image_ori_elem;

  // inputs image, input paramenters
  FP* image;                          // input image
  int Nr,Nc;                          // IMAGE nbr of rows/cols/elements
  long Ne;

  // algorithm parameters
  int niter;                          // nbr of iterations
  FP lambda;                          // update step size

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

  // image size
  int mem_size;

  int blocks_x;
  int blocks_work_size, blocks_work_size2;
  size_t global_work_size, global_work_size2;
  size_t local_work_size;
  int no;
  int mul;
  FP total;
  FP total2;
  FP meanROI;
  FP meanROI2;
  FP varROI;
  FP q0sqr;

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
  image_ori = (FP*)malloc(sizeof(FP) * image_ori_elem);

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

  image = (FP*)malloc(sizeof(FP) * Ne);

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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // allocate memory for derivatives
  FP *d_dN = sycl::malloc_device<FP>(Ne, q);
  FP *d_dS = sycl::malloc_device<FP>(Ne, q);
  FP *d_dW = sycl::malloc_device<FP>(Ne, q);
  FP *d_dE = sycl::malloc_device<FP>(Ne, q);

  // allocate memory for entire IMAGE 
  mem_size = sizeof(FP) * Ne; // get the size of float representation of input IMAGE
  FP *d_I = sycl::malloc_device<FP>(Ne, q);

  // allocate memory for partial sums 
  FP *d_sums = sycl::malloc_device<FP>(Ne, q);
  FP *d_sums2 = sycl::malloc_device<FP>(Ne, q);

  // allocate memory for coordinates 
  int *d_iN = sycl::malloc_device<int>(Nr, q);
  q.memcpy(d_iN, iN, mem_size_i);

  int *d_iS = sycl::malloc_device<int>(Nr, q);
  q.memcpy(d_iS, iS, mem_size_i);

  int *d_jE = sycl::malloc_device<int>(Nc, q);
  q.memcpy(d_jE, jE, mem_size_j);

  int *d_jW = sycl::malloc_device<int>(Nc, q);
  q.memcpy(d_jW, jW, mem_size_j);

  // allocate memory for coefficient 
  FP *d_c = sycl::malloc_device<FP>(Ne, q);

  // threads
  local_work_size = NUMBER_THREADS;

  // workgroups
  blocks_x = Ne/(int)local_work_size;
  if (Ne % (int)local_work_size != 0){ // compensate for division remainder above by adding one grid
    blocks_x = blocks_x + 1;
  }
  blocks_work_size = blocks_x;
  global_work_size = blocks_work_size * local_work_size; // define the number of blocks in the grid

  time5 = get_time();

  //================================================================================80
  //   COPY INPUT TO GPU
  //================================================================================80
  q.memcpy(d_I, image, mem_size).wait();

  time6 = get_time();

  sycl::range<1> gws (global_work_size);
  sycl::range<1> lws (local_work_size);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class extract>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      #include "kernel_extract.sycl"
    });
  }).wait();

  time7 = get_time();

  for (iter=0; iter<niter; iter++){ // do for the number of iterations input parameter
    // Prepare kernel
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class prepare>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        #include "kernel_prepare.sycl"
      });
    });

    blocks_work_size2 = blocks_work_size;  // original number of blocks
    global_work_size2 = global_work_size;
    no = Ne;  // original number of sum elements
    mul = 1;  // original multiplier

    while(blocks_work_size2 != 0){

      sycl::range<1> gws2 (global_work_size2);

      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<FP, 1> d_psum (lws, cgh);
        sycl::local_accessor<FP, 1> d_psum2 (lws, cgh);
        cgh.parallel_for<class reduce>(
          sycl::nd_range<1>(gws2, lws), [=] (sycl::nd_item<1> item) {
          #include "kernel_reduce.sycl"
        });
      }).wait();

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
        global_work_size2 = blocks_work_size2 * (int)local_work_size;
      }
    } // while

    q.memcpy(&total, d_sums, sizeof(FP));
    q.memcpy(&total2, d_sums2, sizeof(FP));

    q.wait();

    // calculate statistics

    meanROI  = total / (FP)(NeROI); // gets mean (average) value of element in ROI
    meanROI2 = meanROI * meanROI;
    varROI = (total2 / (FP)(NeROI)) - meanROI2; // gets variance of ROI
    q0sqr = varROI / meanROI2; // gets standard deviation of ROI

    // set arguments that were uptaded in this loop
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class srad>(
        sycl::nd_range<1>(gws, lws) , [=] (sycl::nd_item<1> item) {
        #include "kernel_srad.sycl"
      });
    });

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class srad2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        #include "kernel_srad2.sycl"
      });
    });
  }

  q.wait();

  time8 = get_time();

  //   Compress Kernel - SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class compress>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      #include "kernel_compress.sycl"
    });
  }).wait();

  time9 = get_time();

  q.memcpy(image, d_I, mem_size).wait();

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

  free(image_ori);
  free(image);
  free(iN);
  free(iS);
  free(jW);
  free(jE);

  sycl::free(d_I, q);
  sycl::free(d_c, q);
  sycl::free(d_iN, q);
  sycl::free(d_iS, q);
  sycl::free(d_jE, q);
  sycl::free(d_jW, q);
  sycl::free(d_dN, q);
  sycl::free(d_dS, q);
  sycl::free(d_dE, q);
  sycl::free(d_dW, q);
  sycl::free(d_sums, q);
  sycl::free(d_sums2, q);

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
}
