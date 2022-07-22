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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);

  const property_list props = property::buffer::use_host_ptr();
  buffer<FP,1> d_dN (Ne);
  buffer<FP,1> d_dS (Ne);
  buffer<FP,1> d_dW (Ne);
  buffer<FP,1> d_dE (Ne);
  buffer<FP,1> d_c (Ne);
  buffer<FP,1> d_sums (Ne);
  buffer<FP,1> d_sums2 (Ne);
  buffer<FP,1> d_I (Ne);
  buffer<int,1> d_iN (iN, Nr, props);
  buffer<int,1> d_iS (iS, Nr, props);
  buffer<int,1> d_jE (jE, Nc, props);
  buffer<int,1> d_jW (jW, Nc, props);

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
  q.submit([&](handler& cgh) {
    auto acc = d_I.get_access<sycl_discard_write>(cgh);
    cgh.copy(image, acc);
  }).wait();

  time6 = get_time();

  range<1> gws (global_work_size); 
  range<1> lws (local_work_size); 

  q.submit([&](handler& cgh) {
    auto d_I_acc = d_I.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class extract>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      #include "kernel_extract.sycl"
    });
  }).wait();

  time7 = get_time();

  for (iter=0; iter<niter; iter++){ // do for the number of iterations input parameter
    // Prepare kernel
    q.submit([&](handler& cgh) {
      auto d_I_acc = d_I.get_access<sycl_read>(cgh);
      auto d_sums_acc = d_sums.get_access<sycl_write>(cgh);
      auto d_sums2_acc = d_sums2.get_access<sycl_write>(cgh);  // updated every iteration
      cgh.parallel_for<class prepare>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        #include "kernel_prepare.sycl"
      });
    });

    blocks_work_size2 = blocks_work_size;  // original number of blocks
    global_work_size2 = global_work_size;
    no = Ne;  // original number of sum elements
    mul = 1;  // original multiplier

    while(blocks_work_size2 != 0){

      range<1> gws2 (global_work_size2); 

      q.submit([&](handler& cgh) {
        auto d_sums_acc = d_sums.get_access<sycl_read_write>(cgh);
        auto d_sums2_acc = d_sums2.get_access<sycl_read_write>(cgh);  // updated every iteration
        accessor <FP, 1, sycl_read_write, access::target::local> d_psum (NUMBER_THREADS, cgh);
        accessor <FP, 1, sycl_read_write, access::target::local> d_psum2 (NUMBER_THREADS, cgh);
        cgh.parallel_for<class reduce>(nd_range<1>(gws2, lws), [=] (nd_item<1> item) {
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

    q.submit([&] (handler &h) {
      auto d_sums_acc = d_sums.get_access<sycl_read>(h, range<1>(1));
      h.copy(d_sums_acc, &total); 
    });

    q.submit([&] (handler &h) {
      auto d_sums_acc = d_sums2.get_access<sycl_read>(h, range<1>(1));
      h.copy(d_sums_acc, &total2); 
    });

    q.wait();
     
    // calculate statistics

    meanROI  = total / (FP)(NeROI); // gets mean (average) value of element in ROI
    meanROI2 = meanROI * meanROI;
    varROI = (total2 / (FP)(NeROI)) - meanROI2; // gets variance of ROI                
    q0sqr = varROI / meanROI2; // gets standard deviation of ROI

    // set arguments that were uptaded in this loop
    q.submit([&](handler& cgh) {
      auto d_iN_acc = d_iN.get_access<sycl_read>(cgh);
      auto d_iS_acc = d_iS.get_access<sycl_read>(cgh);
      auto d_jW_acc = d_jW.get_access<sycl_read>(cgh);
      auto d_jE_acc = d_jE.get_access<sycl_read>(cgh);
      auto d_dN_acc = d_dN.get_access<sycl_write>(cgh);
      auto d_dS_acc = d_dS.get_access<sycl_write>(cgh);
      auto d_dW_acc = d_dW.get_access<sycl_write>(cgh);
      auto d_dE_acc = d_dE.get_access<sycl_write>(cgh);
      auto d_c_acc = d_c.get_access<sycl_write>(cgh);
      auto d_I_acc = d_I.get_access<sycl_read>(cgh);

      cgh.parallel_for<class srad>(nd_range<1>(gws, lws) , [=] (nd_item<1> item) {
        #include "kernel_srad.sycl"
      });
    });

    q.submit([&](handler& cgh) {
      //auto d_iN_acc = d_iN.get_access<sycl_read>(cgh);
      auto d_iS_acc = d_iS.get_access<sycl_read>(cgh);
      //auto d_jW_acc = d_jW.get_access<sycl_read>(cgh);
      auto d_jE_acc = d_jE.get_access<sycl_read>(cgh);
      auto d_dN_acc = d_dN.get_access<sycl_read>(cgh);
      auto d_dS_acc = d_dS.get_access<sycl_read>(cgh);
      auto d_dW_acc = d_dW.get_access<sycl_read>(cgh);
      auto d_dE_acc = d_dE.get_access<sycl_read>(cgh);
      auto d_c_acc = d_c.get_access<sycl_read>(cgh);
      auto d_I_acc = d_I.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class srad2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        #include "kernel_srad2.sycl"
      });
    });
  }

  q.wait();

  time8 = get_time();

  //   Compress Kernel - SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS

  q.submit([&](handler& cgh) {
    auto d_I_acc = d_I.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class compress>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      #include "kernel_compress.sycl"
    });
  }).wait();

  time9 = get_time();

  q.submit([&](handler& cgh) {
    auto acc = d_I.get_access<sycl_read>(cgh);
      cgh.copy(acc, image);
  }).wait();

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
