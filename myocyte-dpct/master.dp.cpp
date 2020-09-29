#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
//=====================================================================
//	MAIN FUNCTION
//=====================================================================

void master(fp timeinst, fp *initvalu, fp *parameter, fp *finavalu, fp *com,

            fp *d_initvalu, fp *d_finavalu, fp *d_params, fp *d_com) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

        //=====================================================================
	//	VARIABLES
	//=====================================================================

	// counters
	int i;

	// offset pointers
	int initvalu_offset_ecc;																// 46 points
	int initvalu_offset_Dyad;															// 15 points
	int initvalu_offset_SL;																// 15 points
	int initvalu_offset_Cyt;																// 15 poitns

	// cuda
 sycl::range<3> threads(1, 1, 1);
 sycl::range<3> blocks(1, 1, 1);

        //=====================================================================
	//	execute ECC&CAM kernel - it runs ECC and CAMs in parallel
	//=====================================================================

	int d_initvalu_mem;
	d_initvalu_mem = EQUATIONS * sizeof(fp);
	int d_finavalu_mem;
	d_finavalu_mem = EQUATIONS * sizeof(fp);
	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(fp);
	int d_com_mem;
	d_com_mem = 3 * sizeof(fp);

#ifdef DEBUG
    for (int i = 0; i < EQUATIONS; i++)
                printf("initvalu %d %f\n", i, initvalu[i]);
  for (int i = 0; i < PARAMETERS; i++)
              printf("params %d %f\n", i, parameter[i]);
    printf("\n");
#endif

 q_ct1.memcpy(d_initvalu, initvalu, d_initvalu_mem).wait();
 q_ct1.memcpy(d_params, parameter, d_params_mem).wait();

 threads[0] = NUMBER_THREADS;
 threads[1] = 1;
 blocks[0] = 2;
 blocks[1] = 1;
 q_ct1.submit([&](sycl::handler &cgh) {
  auto dpct_global_range = blocks * threads;

  cgh.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                         dpct_global_range.get(0)),
          sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
      [=](sycl::nd_item<3> item_ct1) {
       kernel(timeinst, d_initvalu, d_finavalu, d_params, d_com, item_ct1);
      });
 });

 q_ct1.memcpy(finavalu, d_finavalu, d_finavalu_mem).wait();
 q_ct1.memcpy(com, d_com, d_com_mem).wait();

#ifdef DEBUG
    for (int i = 0; i < EQUATIONS; i++)
                printf("finavalu %d %f\n", i, finavalu[i]);
  for (int i = 0; i < 3; i++)
              printf("%f ", com[i]);
    printf("\n");

#endif

	//=====================================================================
	//	FINAL KERNEL
	//=====================================================================

	initvalu_offset_ecc = 0;												// 46 points
	initvalu_offset_Dyad = 46;											// 15 points
	initvalu_offset_SL = 61;											// 15 points
	initvalu_offset_Cyt = 76;												// 15 poitns

	kernel_fin(			initvalu,
								initvalu_offset_ecc,
								initvalu_offset_Dyad,
								initvalu_offset_SL,
								initvalu_offset_Cyt,
								parameter,
								finavalu,
								com[0],
								com[1],
								com[2]);

	//=====================================================================
	//	COMPENSATION FOR NANs and INFs
	//=====================================================================

	for(i=0; i<EQUATIONS; i++){
  if (isnan(finavalu[i])) {
                        finavalu[i] = 0.0001;												// for NAN set rate of change to 0.0001
  } else if (isinf(finavalu[i])) {
                        finavalu[i] = 0.0001;												// for INF set rate of change to 0.0001
		}
	}

}
