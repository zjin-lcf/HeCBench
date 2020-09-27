//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "main.h"
#include "extract_kernel.dp.cpp"
#include "prepare_kernel.dp.cpp"
#include "reduce_kernel.dp.cpp"
#include "srad_kernel.dp.cpp"
#include "srad2_kernel.dp.cpp"
#include "compress_kernel.dp.cpp"
#include "graphics.c"
#include "resize.c"
#include "timer.c"


//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv[]) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

        //================================================================================80
	// 	VARIABLES
	//================================================================================80

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
    fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

    // inputs image, input paramenters
    fp* image;															// input image
    int Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
    int niter;																// nbr of iterations
    fp lambda;															// update step size

    // size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements

    // surrounding pixel indicies
    int *iN,*iS,*jE,*jW;    

    // counters
    int iter;   // primary loop
    long i,j;    // image row/col

	// memory sizes
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	//================================================================================80
	// 	GPU VARIABLES
	//================================================================================80

	// CUDA kernel execution parameters
 sycl::range<3> threads(1, 1, 1);
        int blocks_x;
 sycl::range<3> blocks(1, 1, 1);
 sycl::range<3> blocks2(1, 1, 1);
 sycl::range<3> blocks3(1, 1, 1);

        // memory sizes
	int mem_size;															// matrix memory size

	// HOST
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	// DEVICE
	fp* d_sums;															// partial sum
	fp* d_sums2;
	int* d_iN;
	int* d_iS;
	int* d_jE;
	int* d_jW;
	fp* d_dN; 
	fp* d_dS; 
	fp* d_dW; 
	fp* d_dE;
	fp* d_I;																// input IMAGE on DEVICE
	fp* d_c;

	time1 = get_time();

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	if(argc != 5){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
	}

	time2 = get_time();

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80

    // read image
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

	if ( !read_graphics(	"../data/srad/image.pgm",
								image_ori,
								image_ori_rows,
								image_ori_cols,
								1) ) 
    return -1;

	time3 = get_time();

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80

	Ne = Nr*Nc;

	image = (fp*)malloc(sizeof(fp) * Ne);

	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);

	time4 = get_time();

	//================================================================================80
	// 	SETUP
	//================================================================================80

    r1     = 0;											// top row index of ROI
    r2     = Nr - 1;									// bottom row index of ROI
    c1     = 0;											// left column index of ROI
    c2     = Nc - 1;									// right column index of ROI

	// ROI image size
	NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size

	// allocate variables for surrounding pixels
	mem_size_i = sizeof(int) * Nr;											//
	iN = (int *)malloc(mem_size_i) ;										// north surrounding element
	iS = (int *)malloc(mem_size_i) ;										// south surrounding element
	mem_size_j = sizeof(int) * Nc;											//
	jW = (int *)malloc(mem_size_j) ;										// west surrounding element
	jE = (int *)malloc(mem_size_j) ;										// east surrounding element

	// N/S/W/E indices of surrounding pixels (every element of IMAGE)
	for (i=0; i<Nr; i++) {
		iN[i] = i-1;														// holds index of IMAGE row above
		iS[i] = i+1;														// holds index of IMAGE row below
	}
	for (j=0; j<Nc; j++) {
		jW[j] = j-1;														// holds index of IMAGE column on the left
		jE[j] = j+1;														// holds index of IMAGE column on the right
	}

	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
	iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
	iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1 
	jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
	jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

	//================================================================================80
	// 	GPU SETUP
	//================================================================================80

	// allocate memory for entire IMAGE on DEVICE
	mem_size = sizeof(fp) * Ne;																		// get the size of float representation of input IMAGE
 dpct::dpct_malloc((void **)&d_I, mem_size); //

        // allocate memory for coordinates on DEVICE
 dpct::dpct_malloc((void **)&d_iN, mem_size_i);                 //
 dpct::dpct_memcpy(d_iN, iN, mem_size_i, dpct::host_to_device); //
 dpct::dpct_malloc((void **)&d_iS, mem_size_i);                 //
 dpct::dpct_memcpy(d_iS, iS, mem_size_i, dpct::host_to_device); //
 dpct::dpct_malloc((void **)&d_jE, mem_size_j);                 //
 dpct::dpct_memcpy(d_jE, jE, mem_size_j, dpct::host_to_device); //
 dpct::dpct_malloc((void **)&d_jW, mem_size_j);                 //
 dpct::dpct_memcpy(d_jW, jW, mem_size_j, dpct::host_to_device); //

        // allocate memory for partial sums on DEVICE
 dpct::dpct_malloc((void **)&d_sums, mem_size);  //
 dpct::dpct_malloc((void **)&d_sums2, mem_size); //

        // allocate memory for derivatives
 dpct::dpct_malloc((void **)&d_dN, mem_size); //
 dpct::dpct_malloc((void **)&d_dS, mem_size); //
 dpct::dpct_malloc((void **)&d_dW, mem_size); //
 dpct::dpct_malloc((void **)&d_dE, mem_size); //

        // allocate memory for coefficient on DEVICE
 dpct::dpct_malloc((void **)&d_c, mem_size); //

        //checkCUDAError("setup");

	//================================================================================80
	// 	KERNEL EXECUTION PARAMETERS
	//================================================================================80

	// all kernels operating on entire matrix
 threads[0] = NUMBER_THREADS; // define the number of threads in the block
 threads[1] = 1;
 blocks_x = Ne / threads[0];
 if (Ne % threads[0] !=
     0) { // compensate for division remainder above by adding one grid
                blocks_x = blocks_x + 1;																	
	}
 blocks[0] = blocks_x; // define the number of blocks in the grid
 blocks[1] = 1;

        time5 = get_time();

	//================================================================================80
	// 	COPY INPUT TO CPU
	//================================================================================80

 dpct::dpct_memcpy(d_I, image, mem_size, dpct::host_to_device);

        time6 = get_time();

	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80

 {
  dpct::buffer_t d_I_buf_ct1 = dpct::get_buffer(d_I);
  q_ct1.submit([&](sycl::handler &cgh) {
   auto d_I_acc_ct1 =
       d_I_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

   auto dpct_global_range = blocks * threads;

   cgh.parallel_for(
       sycl::nd_range<3>(
           sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                          dpct_global_range.get(0)),
           sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
       [=](sycl::nd_item<3> item_ct1) {
        extract(Ne, (float *)(&d_I_acc_ct1[0]), item_ct1);
       });
  });
 }

        //checkCUDAError("extract");

	time7 = get_time();

	//================================================================================80
	// 	COMPUTATION
	//================================================================================80

	// printf("iterations: ");

	// execute main loop
	for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

	// printf("%d ", iter);
	// fflush(NULL);

		// execute square kernel
  {
   dpct::buffer_t d_I_buf_ct1 = dpct::get_buffer(d_I);
   dpct::buffer_t d_sums_buf_ct2 = dpct::get_buffer(d_sums);
   dpct::buffer_t d_sums2_buf_ct3 = dpct::get_buffer(d_sums2);
   q_ct1.submit([&](sycl::handler &cgh) {
    auto d_I_acc_ct1 =
        d_I_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
    auto d_sums_acc_ct2 =
        d_sums_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
    auto d_sums2_acc_ct3 =
        d_sums2_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);

    auto dpct_global_range = blocks * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
         prepare(Ne, (const float *)(&d_I_acc_ct1[0]),
                 (float *)(&d_sums_acc_ct2[0]), (float *)(&d_sums2_acc_ct3[0]),
                 item_ct1);
        });
   });
  }

                //checkCUDAError("prepare");

		// performs subsequent reductions of sums
  blocks2[0] = blocks[0]; // original number of blocks
  blocks2[1] = blocks[1];
                no = Ne;														// original number of sum elements
		mul = 1;														// original multiplier

  while (blocks2[0] != 0) {

                        //checkCUDAError("before reduce");

			// run kernel
   {
    dpct::buffer_t d_sums_buf_ct3 = dpct::get_buffer(d_sums);
    dpct::buffer_t d_sums2_buf_ct4 = dpct::get_buffer(d_sums2);
    q_ct1.submit([&](sycl::handler &cgh) {
     sycl::accessor<fp, 1, sycl::access::mode::read_write,
                    sycl::access::target::local>
         d_psum_acc_ct1(sycl::range<1>(256 /*NUMBER_THREADS*/), cgh);
     sycl::accessor<fp, 1, sycl::access::mode::read_write,
                    sycl::access::target::local>
         d_psum2_acc_ct1(sycl::range<1>(256 /*NUMBER_THREADS*/), cgh);
     auto d_sums_acc_ct3 =
         d_sums_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
     auto d_sums2_acc_ct4 =
         d_sums2_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);

     auto dpct_global_range = blocks2 * threads;

     cgh.parallel_for(
         sycl::nd_range<3>(
             sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                            dpct_global_range.get(0)),
             sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
         [=](sycl::nd_item<3> item_ct1) {
          reduce(Ne, no, mul, (float *)(&d_sums_acc_ct3[0]),
                 (float *)(&d_sums2_acc_ct4[0]), item_ct1,
                 d_psum_acc_ct1.get_pointer(), d_psum2_acc_ct1.get_pointer());
         });
    });
   }

                        //checkCUDAError("reduce");

			// update execution parameters
   no = blocks2[0]; // get current number of elements
   if (blocks2[0] == 1) {
    blocks2[0] = 0;
                        }
			else{
				mul = mul * NUMBER_THREADS;									// update the increment
    blocks_x = blocks2[0] / threads[0]; // number of blocks
    if (blocks2[0] % threads[0] !=
        0) { // compensate for division remainder above by adding one grid
                                        blocks_x = blocks_x + 1;
				}
    blocks2[0] = blocks_x;
    blocks2[1] = 1;
                        }

			//checkCUDAError("after reduce");

		}

		//checkCUDAError("before copy sum");

		// copy total sums to device
		mem_size_single = sizeof(fp) * 1;
  dpct::dpct_memcpy(&total, d_sums, mem_size_single, dpct::device_to_host);
  dpct::dpct_memcpy(&total2, d_sums2, mem_size_single, dpct::device_to_host);

                //checkCUDAError("copy sum");

		// calculate statistics
		meanROI	= total / fp(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (total2 / fp(NeROI)) - meanROI2;						// gets variance of ROI								
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		// execute srad kernel
  {
   dpct::buffer_t d_iN_buf_ct4 = dpct::get_buffer(d_iN);
   dpct::buffer_t d_iS_buf_ct5 = dpct::get_buffer(d_iS);
   dpct::buffer_t d_jE_buf_ct6 = dpct::get_buffer(d_jE);
   dpct::buffer_t d_jW_buf_ct7 = dpct::get_buffer(d_jW);
   dpct::buffer_t d_dN_buf_ct8 = dpct::get_buffer(d_dN);
   dpct::buffer_t d_dS_buf_ct9 = dpct::get_buffer(d_dS);
   dpct::buffer_t d_dW_buf_ct10 = dpct::get_buffer(d_dW);
   dpct::buffer_t d_dE_buf_ct11 = dpct::get_buffer(d_dE);
   dpct::buffer_t d_c_buf_ct13 = dpct::get_buffer(d_c);
   dpct::buffer_t d_I_buf_ct14 = dpct::get_buffer(d_I);
   q_ct1.submit([&](sycl::handler &cgh) {
    auto d_iN_acc_ct4 =
        d_iN_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
    auto d_iS_acc_ct5 =
        d_iS_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
    auto d_jE_acc_ct6 =
        d_jE_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
    auto d_jW_acc_ct7 =
        d_jW_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dN_acc_ct8 =
        d_dN_buf_ct8.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dS_acc_ct9 =
        d_dS_buf_ct9.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dW_acc_ct10 =
        d_dW_buf_ct10.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dE_acc_ct11 =
        d_dE_buf_ct11.get_access<sycl::access::mode::read_write>(cgh);
    auto d_c_acc_ct13 =
        d_c_buf_ct13.get_access<sycl::access::mode::read_write>(cgh);
    auto d_I_acc_ct14 =
        d_I_buf_ct14.get_access<sycl::access::mode::read_write>(cgh);

    auto dpct_global_range = blocks * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
         srad(lambda, Nr, Nc, Ne, (const int *)(&d_iN_acc_ct4[0]),
              (const int *)(&d_iS_acc_ct5[0]), (const int *)(&d_jE_acc_ct6[0]),
              (const int *)(&d_jW_acc_ct7[0]), (float *)(&d_dN_acc_ct8[0]),
              (float *)(&d_dS_acc_ct9[0]), (float *)(&d_dW_acc_ct10[0]),
              (float *)(&d_dE_acc_ct11[0]), q0sqr, (float *)(&d_c_acc_ct13[0]),
              (const float *)(&d_I_acc_ct14[0]), item_ct1);
        });
   });
  } // output image

                //checkCUDAError("srad");

		// execute srad2 kernel
  {
   dpct::buffer_t d_iN_buf_ct4 = dpct::get_buffer(d_iN);
   dpct::buffer_t d_iS_buf_ct5 = dpct::get_buffer(d_iS);
   dpct::buffer_t d_jE_buf_ct6 = dpct::get_buffer(d_jE);
   dpct::buffer_t d_jW_buf_ct7 = dpct::get_buffer(d_jW);
   dpct::buffer_t d_dN_buf_ct8 = dpct::get_buffer(d_dN);
   dpct::buffer_t d_dS_buf_ct9 = dpct::get_buffer(d_dS);
   dpct::buffer_t d_dW_buf_ct10 = dpct::get_buffer(d_dW);
   dpct::buffer_t d_dE_buf_ct11 = dpct::get_buffer(d_dE);
   dpct::buffer_t d_c_buf_ct12 = dpct::get_buffer(d_c);
   dpct::buffer_t d_I_buf_ct13 = dpct::get_buffer(d_I);
   q_ct1.submit([&](sycl::handler &cgh) {
    auto d_iN_acc_ct4 =
        d_iN_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
    auto d_iS_acc_ct5 =
        d_iS_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
    auto d_jE_acc_ct6 =
        d_jE_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
    auto d_jW_acc_ct7 =
        d_jW_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dN_acc_ct8 =
        d_dN_buf_ct8.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dS_acc_ct9 =
        d_dS_buf_ct9.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dW_acc_ct10 =
        d_dW_buf_ct10.get_access<sycl::access::mode::read_write>(cgh);
    auto d_dE_acc_ct11 =
        d_dE_buf_ct11.get_access<sycl::access::mode::read_write>(cgh);
    auto d_c_acc_ct12 =
        d_c_buf_ct12.get_access<sycl::access::mode::read_write>(cgh);
    auto d_I_acc_ct13 =
        d_I_buf_ct13.get_access<sycl::access::mode::read_write>(cgh);

    auto dpct_global_range = blocks * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
         srad2(lambda, Nr, Nc, Ne, (const int *)(&d_iN_acc_ct4[0]),
               (const int *)(&d_iS_acc_ct5[0]), (const int *)(&d_jE_acc_ct6[0]),
               (const int *)(&d_jW_acc_ct7[0]),
               (const float *)(&d_dN_acc_ct8[0]),
               (const float *)(&d_dS_acc_ct9[0]),
               (const float *)(&d_dW_acc_ct10[0]),
               (const float *)(&d_dE_acc_ct11[0]),
               (const float *)(&d_c_acc_ct12[0]), (float *)(&d_I_acc_ct13[0]),
               item_ct1);
        });
   });
  } // output image

                //checkCUDAError("srad2");

	}

	// printf("\n");

	time8 = get_time();

	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80

 {
  dpct::buffer_t d_I_buf_ct1 = dpct::get_buffer(d_I);
  q_ct1.submit([&](sycl::handler &cgh) {
   auto d_I_acc_ct1 =
       d_I_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

   auto dpct_global_range = blocks * threads;

   cgh.parallel_for(
       sycl::nd_range<3>(
           sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                          dpct_global_range.get(0)),
           sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
       [=](sycl::nd_item<3> item_ct1) {
        compress(Ne, (float *)(&d_I_acc_ct1[0]), item_ct1);
       });
  });
 }

        //checkCUDAError("compress");

	time9 = get_time();

	//================================================================================80
	// 	COPY RESULTS BACK TO CPU
	//================================================================================80

 dpct::dpct_memcpy(image, d_I, mem_size, dpct::device_to_host);

        //checkCUDAError("copy back");

	time10 = get_time();

	//================================================================================80
	// 	WRITE IMAGE AFTER PROCESSING
	//================================================================================80

	write_graphics(	"image_out.pgm",
					image,
					Nr,
					Nc,
					1,
					255);

	time11 = get_time();

	//================================================================================80
	//	DEALLOCATE
	//================================================================================80

	free(image_ori);
	free(image);
	free(iN); 
	free(iS); 
	free(jW); 
	free(jE);

 dpct::dpct_free(d_I);
 dpct::dpct_free(d_c);
 dpct::dpct_free(d_iN);
 dpct::dpct_free(d_iS);
 dpct::dpct_free(d_jE);
 dpct::dpct_free(d_jW);
 dpct::dpct_free(d_dN);
 dpct::dpct_free(d_dS);
 dpct::dpct_free(d_dE);
 dpct::dpct_free(d_dW);
 dpct::dpct_free(d_sums);
 dpct::dpct_free(d_sums2);

        time12 = get_time();

	//================================================================================80
	//	DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n", 														(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n", 										(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n", 												(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n", 														(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n", 					(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n", 												(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n", 														(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COMPUTE\n", 																(float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n", 														(float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n", 												(float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n", 												(float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : FREE MEMORY\n", 															(float) (time12-time11) / 1000000, (float) (time12-time11) / (float) (time12-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																					(float) (time12-time0) / 1000000);

}

//====================================================================================================100
//	END OF FILE
//====================================================================================================100
