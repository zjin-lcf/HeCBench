#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "./../main.h" // (in main directory)            needed to recognized input parameters
#include "./../util/avi/avilib.h"          // (in directory)              needed by avi functions
#include "./../util/avi/avimod.h"          // (in directory)              needed by avi functions

// CUDA kernel
#include "kernel.h"

  void 
kernel_gpu_wrapper(  params_common common,
    int* endoRow,
    int* endoCol,
    int* tEndoRowLoc,
    int* tEndoColLoc,
    int* epiRow,
    int* epiCol,
    int* tEpiRowLoc,
    int* tEpiColLoc,
    avi_t* frames)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

  // common
  //printf("tSize is %d, sSize is %d\n", common.tSize, common.sSize);
  common.in_rows = common.tSize + 1 + common.tSize;
  common.in_cols = common.in_rows;
  common.in_elem = common.in_rows * common.in_cols;
  common.in_mem = sizeof(fp) * common.in_elem;

  //==================================================50
  // endo points templates
  //==================================================50

  fp* d_endoT;
 d_endoT =
     (float *)sycl::malloc_device(common.in_mem * common.endoPoints, q_ct1);
  //printf("%d\n", common.in_elem * common.endoPoints);

  //==================================================50
  // epi points templates
  //==================================================50

  fp* d_epiT;
 d_epiT = (float *)sycl::malloc_device(common.in_mem * common.epiPoints, q_ct1);

  //====================================================================================================100
  //   AREA AROUND POINT    FROM  FRAME  (LOCAL)
  //====================================================================================================100

  // common
  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(fp) * common.in2_elem;

  fp* d_in2;
 d_in2 = (float *)sycl::malloc_device(common.in2_mem * common.allPoints, q_ct1);
  //printf("%d\n", common.in2_elem * common.allPoints);

  //====================================================================================================100
  //   CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.conv_rows = common.in_rows + common.in2_rows - 1;                        // number of rows in I
  common.conv_cols = common.in_cols + common.in2_cols - 1;                        // number of columns in I
  common.conv_elem = common.conv_rows * common.conv_cols;                          // number of elements
  common.conv_mem = sizeof(fp) * common.conv_elem;
  common.ioffset = 0;
  common.joffset = 0;

  // unique
  fp* d_conv;
 d_conv =
     (float *)sycl::malloc_device(common.conv_mem * common.allPoints, q_ct1);

  //====================================================================================================100
  //   CUMULATIVE SUM  (LOCAL)
  //====================================================================================================100

  //==================================================50
  //   PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
  //==================================================50

  // common
  common.in2_pad_add_rows = common.in_rows;
  common.in2_pad_add_cols = common.in_cols;

  common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
  common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
  common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
  common.in2_pad_cumv_mem = sizeof(fp) * common.in2_pad_cumv_elem;

  // unique
  //buffer<fp,1> d_in2_pad_cumv(common.in2_pad_cumv_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_elem * common.allPoints);
  fp* d_in2_pad_cumv;
 d_in2_pad_cumv = (float *)sycl::malloc_device(
     common.in2_pad_cumv_mem * common.allPoints, q_ct1);

  //==================================================50
  //   SELECTION
  //==================================================50

  // common
  common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;                          // (1 to n+1)
  common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
  common.in2_pad_cumv_sel_collow = 1;
  common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
  common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
  common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
  common.in2_pad_cumv_sel_mem = sizeof(fp) * common.in2_pad_cumv_sel_elem;

  // unique
  //buffer<fp,1> d_in2_pad_cumv_sel(common.in2_pad_cumv_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_sel_elem * common.allPoints);
  fp* d_in2_pad_cumv_sel;
 d_in2_pad_cumv_sel = (float *)sycl::malloc_device(
     common.in2_pad_cumv_sel_mem * common.allPoints, q_ct1);

  //==================================================50
  //   SELECTION  2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
  //==================================================50

  // common
  common.in2_pad_cumv_sel2_rowlow = 1;
  common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
  common.in2_pad_cumv_sel2_collow = 1;
  common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
  common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
  common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
  common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
  common.in2_sub_cumh_mem = sizeof(fp) * common.in2_sub_cumh_elem;

  // unique
  //buffer<fp,1> d_in2_sub_cumh(common.in2_sub_cumh_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_elem * common.allPoints);
  fp* d_in2_sub_cumh;
 d_in2_sub_cumh = (float *)sycl::malloc_device(
     common.in2_sub_cumh_mem * common.allPoints, q_ct1);

  //==================================================50
  //   SELECTION
  //==================================================50

  // common
  common.in2_sub_cumh_sel_rowlow = 1;
  common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
  common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
  common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
  common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
  common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
  common.in2_sub_cumh_sel_mem = sizeof(fp) * common.in2_sub_cumh_sel_elem;

  // unique
  //buffer<fp,1> d_in2_sub_cumh_sel(common.in2_sub_cumh_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_sel_elem * common.allPoints);
  fp* d_in2_sub_cumh_sel;
 d_in2_sub_cumh_sel = (float *)sycl::malloc_device(
     common.in2_sub_cumh_sel_mem * common.allPoints, q_ct1);

  //==================================================50
  //  SELECTION 2, SUBTRACTION
  //==================================================50

  // common
  common.in2_sub_cumh_sel2_rowlow = 1;
  common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel2_collow = 1;
  common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
  common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
  common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
  common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
  common.in2_sub2_mem = sizeof(fp) * common.in2_sub2_elem;

  // unique
  //buffer<fp,1> d_in2_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);
  fp* d_in2_sub2;
 d_in2_sub2 = (float *)sycl::malloc_device(
     common.in2_sub2_mem * common.allPoints, q_ct1);

  //====================================================================================================100
  //  CUMULATIVE SUM 2  (LOCAL)
  //====================================================================================================100

  //==================================================50
  //  MULTIPLICATION
  //==================================================50

  // common
  common.in2_sqr_rows = common.in2_rows;
  common.in2_sqr_cols = common.in2_cols;
  common.in2_sqr_elem = common.in2_elem;
  common.in2_sqr_mem = common.in2_mem;

  // unique
  //buffer<fp,1> d_in2_sqr(common.in2_elem * common.allPoints);
  //printf("%d\n", common.in2_elem * common.allPoints);
  fp* d_in2_sqr;
 d_in2_sqr =
     (float *)sycl::malloc_device(common.in2_sqr_mem * common.allPoints, q_ct1);

  //==================================================50
  //  SELECTION 2, SUBTRACTION
  //==================================================50

  // common
  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem = common.in2_sub2_mem;

  // unique
  //buffer<fp,1> d_in2_sqr_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);
  fp* d_in2_sqr_sub2;
 d_in2_sqr_sub2 = (float *)sycl::malloc_device(
     common.in2_sqr_sub2_mem * common.allPoints, q_ct1);

  //====================================================================================================100
  //  FINAL  (LOCAL)
  //====================================================================================================100

  // common
  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem = common.in_mem;

  // unique
  //buffer<fp,1> d_in_sqr(common.in_elem * common.allPoints);
  //printf("%d\n", common.in_elem * common.allPoints);
  fp* d_in_sqr;
 d_in_sqr =
     (float *)sycl::malloc_device(common.in_sqr_mem * common.allPoints, q_ct1);

  //====================================================================================================100
  //  TEMPLATE MASK CREATE  (LOCAL)
  //====================================================================================================100

  // common
  common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
  common.tMask_cols = common.tMask_rows;
  common.tMask_elem = common.tMask_rows * common.tMask_cols;
  common.tMask_mem = sizeof(fp) * common.tMask_elem;

  // unique
  //buffer<fp,1> d_tMask(common.tMask_elem * common.allPoints);
  //printf("%d\n", common.tMask_elem * common.allPoints);
  fp* d_tMask;
 d_tMask =
     (float *)sycl::malloc_device(common.tMask_mem * common.allPoints, q_ct1);

  //====================================================================================================100
  //  POINT MASK INITIALIZE  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_mem = sizeof(fp) * common.mask_elem;

  //====================================================================================================100
  //  MASK CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_conv_rows = common.tMask_rows;                        // number of rows in I
  common.mask_conv_cols = common.tMask_cols;                        // number of columns in I
  common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;                        // number of elements
  common.mask_conv_mem = sizeof(fp) * common.mask_conv_elem;
  common.mask_conv_ioffset = (common.mask_rows-1)/2;
  if((common.mask_rows-1) % 2 > 0.5){
    common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
  }
  common.mask_conv_joffset = (common.mask_cols-1)/2;
  if((common.mask_cols-1) % 2 > 0.5){
    common.mask_conv_joffset = common.mask_conv_joffset + 1;
  }


  //printf("common.endPoints=%d\n", common.endoPoints); // 20 
  //printf("common.epiPoints=%d\n", common.epiPoints); // 31
  //printf("common.in_elem=%d\n", common.in_elem);
  //printf("common.endo_mem=%d\n", common.endo_mem); // 80
  //printf("common.epi_mem=%d\n", common.epi_mem); // 124
  //
  //buffer<params_common,1> d_common(&common, 1, props); // range is 1 ?
  //buffer<int,1> d_endoRow(endoRow, common.endoPoints, props);
  //d_endoRow.set_final_data(nullptr);
  //buffer<int,1> d_endoCol(endoCol, common.endoPoints, props);
  //d_endoCol.set_final_data(nullptr);
  //buffer<int,1> d_tEndoRowLoc(tEndoRowLoc, common.endoPoints * common.no_frames, props);
  //buffer<int,1> d_tEndoColLoc(tEndoColLoc, common.endoPoints * common.no_frames, props);
  //buffer<int,1> d_epiRow(epiRow, common.epiPoints, props);
  //d_epiRow.set_final_data(nullptr);
  //buffer<int,1> d_epiCol(epiCol, common.epiPoints, props);
  //d_epiCol.set_final_data(nullptr);
  //buffer<int,1> d_tEpiRowLoc(tEpiRowLoc, common.epiPoints * common.no_frames, props);
  //buffer<int,1> d_tEpiColLoc(tEpiColLoc, common.epiPoints * common.no_frames, props);

  int* d_endoRow;
 d_endoRow = (int *)sycl::malloc_device(common.endo_mem, q_ct1);
 q_ct1.memcpy(d_endoRow, endoRow, common.endo_mem).wait();

  int* d_endoCol;
 d_endoCol = (int *)sycl::malloc_device(common.endo_mem, q_ct1);
 q_ct1.memcpy(d_endoCol, endoCol, common.endo_mem).wait();

  int* d_tEndoRowLoc;
  int* d_tEndoColLoc;
 d_tEndoRowLoc =
     (int *)sycl::malloc_device(common.endo_mem * common.no_frames, q_ct1);
 q_ct1.memcpy(d_tEndoRowLoc, tEndoRowLoc, common.endo_mem * common.no_frames)
     .wait();
 d_tEndoColLoc =
     (int *)sycl::malloc_device(common.endo_mem * common.no_frames, q_ct1);
 q_ct1.memcpy(d_tEndoColLoc, tEndoColLoc, common.endo_mem * common.no_frames)
     .wait();

  int* d_epiRow;
  int* d_epiCol;
 d_epiRow = (int *)sycl::malloc_device(common.epi_mem, q_ct1);
 q_ct1.memcpy(d_epiRow, epiRow, common.epi_mem).wait();
 d_epiCol = (int *)sycl::malloc_device(common.epi_mem, q_ct1);
 q_ct1.memcpy(d_epiCol, epiCol, common.epi_mem).wait();

  int* d_tEpiRowLoc;
  int* d_tEpiColLoc;
 d_tEpiRowLoc =
     (int *)sycl::malloc_device(common.epi_mem * common.no_frames, q_ct1);
 q_ct1.memcpy(d_tEpiRowLoc, tEpiRowLoc, common.epi_mem * common.no_frames)
     .wait();
 d_tEpiColLoc =
     (int *)sycl::malloc_device(common.epi_mem * common.no_frames, q_ct1);
 q_ct1.memcpy(d_tEpiColLoc, tEpiColLoc, common.epi_mem * common.no_frames)
     .wait();

  //buffer<fp,1> d_mask_conv(common.mask_conv_elem * common.allPoints);
  //d_mask_conv.set_final_data(nullptr);
  fp* d_mask_conv;
 d_mask_conv = (float *)sycl::malloc_device(
     common.mask_conv_mem * common.allPoints, q_ct1);

  //printf("%d\n", common.mask_conv_elem * common.allPoints);
  //buffer<fp,1> d_in_mod_temp(common.in_elem * common.allPoints);
  //d_in_mod_temp.set_final_data(nullptr);
  fp* d_in_mod_temp;
 d_in_mod_temp =
     (float *)sycl::malloc_device(common.in_mem * common.allPoints, q_ct1);

  //printf("%d\n", common.in_elem * common.allPoints);
  //buffer<fp,1> d_in_partial_sum(common.in_cols * common.allPoints);
  //d_in_partial_sum.set_final_data(nullptr);

  fp* d_in_partial_sum;
 d_in_partial_sum = (float *)sycl::malloc_device(
     sizeof(fp) * common.in_cols * common.allPoints, q_ct1);

  //printf("%d\n", common.in_cols * common.allPoints);
  //buffer<fp,1> d_in_sqr_partial_sum(common.in_sqr_rows * common.allPoints);
  //d_in_sqr_partial_sum.set_final_data(nullptr);

  fp* d_in_sqr_partial_sum;
 d_in_sqr_partial_sum = (float *)sycl::malloc_device(
     sizeof(fp) * common.in_sqr_rows * common.allPoints, q_ct1);

  //printf("%d\n", common.in_sqr_rows * common.allPoints);
  //buffer<fp,1> d_par_max_val(common.mask_conv_rows * common.allPoints);
  //d_par_max_val.set_final_data(nullptr);

  fp* d_par_max_val;
 d_par_max_val = (float *)sycl::malloc_device(
     sizeof(fp) * common.mask_conv_rows * common.allPoints, q_ct1);

  //printf("%d\n", common.mask_conv_rows * common.allPoints);
  //buffer<int,1> d_par_max_coo( common.mask_conv_rows * common.allPoints);
  //d_par_max_coo.set_final_data(nullptr);

  fp* d_par_max_coo;
 d_par_max_coo = (float *)sycl::malloc_device(
     sizeof(fp) * common.mask_conv_rows * common.allPoints, q_ct1);

  //buffer<fp,1> d_in_final_sum(common.allPoints);
  //d_in_final_sum.set_final_data(nullptr);

  fp* d_in_final_sum;
 d_in_final_sum = sycl::malloc_device<float>(common.allPoints, q_ct1);

  //buffer<fp,1> d_in_sqr_final_sum(common.allPoints);
  //d_in_sqr_final_sum.set_final_data(nullptr);
  fp* d_in_sqr_final_sum;
 d_in_sqr_final_sum = sycl::malloc_device<float>(common.allPoints, q_ct1);

  //buffer<fp,1> d_denomT(common.allPoints);
  //d_denomT.set_final_data(nullptr);

  fp* d_denomT;
 d_denomT = sycl::malloc_device<float>(common.allPoints, q_ct1);

#ifdef TEST_CHECKSUM
  //buffer<fp,1> d_checksum(CHECK);
  //d_checksum.set_final_data(nullptr);
  //printf("%d\n", CHECK);
  fp* checksum = (fp*) malloc (sizeof(fp)*CHECK);
  fp* d_checksum;
 d_checksum = sycl::malloc_device<float>(CHECK, q_ct1);
#endif

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  // All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
 sycl::range<3> threads(NUMBER_THREADS, 1, 1);
 sycl::range<3> grids(common.allPoints, 1, 1);

  printf("frame progress: ");
  fflush(NULL);

  //====================================================================================================100
  //  LAUNCH
  //====================================================================================================100

  // variables
  fp* frame;
  int frame_no;

  //buffer<fp,1> d_frame(common.frame_elem);
  fp* d_frame;
 d_frame = sycl::malloc_device<float>(common.frame_elem, q_ct1);

  for(frame_no=0; frame_no<common.frames_processed; frame_no++) {

    //==================================================50
    //  get and write current frame to GPU buffer
    //==================================================50

    // Extract a cropped version of the first frame from the video file
    frame = get_frame(  frames,                // pointer to video file
        frame_no,              // number of frame that needs to be returned
        0,                  // cropped?
        0,                  // scaled?
        1);                  // converted

    // copy frame to GPU memory
  q_ct1.memcpy(d_frame, frame, sizeof(fp) * common.frame_elem).wait();

    //==================================================50
    //  launch kernel
    //==================================================50
  q_ct1.submit([&](sycl::handler &cgh) {
   auto dpct_global_range = grids * threads;

   cgh.parallel_for(
       sycl::nd_range<3>(
           sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                          dpct_global_range.get(0)),
           sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
       [=](sycl::nd_item<3> item_ct1) {
        hw(frame_no, common, d_frame, d_endoRow, d_endoCol, d_tEndoRowLoc,
           d_tEndoColLoc, d_epiRow, d_epiCol, d_tEpiRowLoc, d_tEpiColLoc,
           d_endoT, d_epiT, d_in2, d_conv, d_in2_pad_cumv, d_in2_pad_cumv_sel,
           d_in2_sub_cumh, d_in2_sub_cumh_sel, d_in2_sub2, d_in2_sqr,
           d_in2_sqr_sub2, d_in_sqr, d_tMask, d_mask_conv, d_in_mod_temp,
           d_in_partial_sum, d_in_sqr_partial_sum, d_par_max_val, d_par_max_coo,
           d_in_final_sum, d_in_sqr_final_sum, d_denomT, 
#ifdef TEST_CHECKSUM
	   d_checksum, 
#endif
	   item_ct1);
       });
  });

    // free frame after each loop iteration, since AVI library allocates memory for every frame fetched
    free(frame);

    //==================================================50
    //  print frame progress
    //==================================================50

    // print frame progress
    printf("%d ", frame_no);
    fflush(NULL);

    //==================================================50
    //  DISPLAY CHECKSUM (TESTING)
    //==================================================50

#ifdef TEST_CHECKSUM
  q_ct1.memcpy(checksum, d_checksum, sizeof(fp) * CHECK).wait();
    printf("CHECKSUM:\n");
    for(int i=0; i<CHECK; i++){
      printf("i=%d checksum=%f\n", i, checksum[i]);
    }
    printf("\n\n");
#endif

  }

 q_ct1.memcpy(tEndoRowLoc, d_tEndoRowLoc, common.endo_mem * common.no_frames)
     .wait();
 q_ct1.memcpy(tEndoColLoc, d_tEndoColLoc, common.endo_mem * common.no_frames)
     .wait();
 q_ct1.memcpy(tEpiRowLoc, d_tEpiRowLoc, common.epi_mem * common.no_frames)
     .wait();
 q_ct1.memcpy(tEpiColLoc, d_tEpiColLoc, common.epi_mem * common.no_frames)
     .wait();

  //====================================================================================================100
  //  PRINT FRAME PROGRESS END
  //====================================================================================================100
#ifdef TEST_CHECKSUM
  free(checksum);
 sycl::free(d_checksum, q_ct1);
#endif
 sycl::free(d_epiT, q_ct1);
 sycl::free(d_in2, q_ct1);
 sycl::free(d_conv, q_ct1);
 sycl::free(d_in2_pad_cumv, q_ct1);
 sycl::free(d_in2_pad_cumv_sel, q_ct1);
 sycl::free(d_in2_sub_cumh, q_ct1);
 sycl::free(d_in2_sub_cumh_sel, q_ct1);
 sycl::free(d_in2_sub2, q_ct1);
 sycl::free(d_in2_sqr, q_ct1);
 sycl::free(d_in2_sqr_sub2, q_ct1);
 sycl::free(d_in_sqr, q_ct1);
 sycl::free(d_tMask, q_ct1);
 sycl::free(d_endoRow, q_ct1);
 sycl::free(d_endoCol, q_ct1);
 sycl::free(d_tEndoRowLoc, q_ct1);
 sycl::free(d_tEndoColLoc, q_ct1);
 sycl::free(d_epiRow, q_ct1);
 sycl::free(d_epiCol, q_ct1);
 sycl::free(d_tEpiRowLoc, q_ct1);
 sycl::free(d_tEpiColLoc, q_ct1);
 sycl::free(d_mask_conv, q_ct1);
 sycl::free(d_in_mod_temp, q_ct1);
 sycl::free(d_in_partial_sum, q_ct1);
 sycl::free(d_in_sqr_partial_sum, q_ct1);
 sycl::free(d_par_max_val, q_ct1);
 sycl::free(d_par_max_coo, q_ct1);
 sycl::free(d_in_final_sum, q_ct1);
 sycl::free(d_in_sqr_final_sum, q_ct1);
 sycl::free(d_denomT, q_ct1);
 sycl::free(d_frame, q_ct1);

  printf("\n");
  fflush(NULL);

}

