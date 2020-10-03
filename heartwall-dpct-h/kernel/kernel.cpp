#define DPCT_USM_LEVEL_NONE
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
 dpct::dpct_malloc((void **)&d_endoT, common.in_mem * common.endoPoints);
  //printf("%d\n", common.in_elem * common.endoPoints);

  //==================================================50
  // epi points templates
  //==================================================50

  fp* d_epiT;
 dpct::dpct_malloc((void **)&d_epiT, common.in_mem * common.epiPoints);

  //====================================================================================================100
  //   AREA AROUND POINT    FROM  FRAME  (LOCAL)
  //====================================================================================================100

  // common
  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(fp) * common.in2_elem;

  fp* d_in2;
 dpct::dpct_malloc((void **)&d_in2, common.in2_mem * common.allPoints);
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
 dpct::dpct_malloc((void **)&d_conv, common.conv_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in2_pad_cumv,
                   common.in2_pad_cumv_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in2_pad_cumv_sel,
                   common.in2_pad_cumv_sel_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in2_sub_cumh,
                   common.in2_sub_cumh_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in2_sub_cumh_sel,
                   common.in2_sub_cumh_sel_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in2_sub2,
                   common.in2_sub2_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in2_sqr, common.in2_sqr_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in2_sqr_sub2,
                   common.in2_sqr_sub2_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_in_sqr, common.in_sqr_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_tMask, common.tMask_mem * common.allPoints);

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
 dpct::dpct_malloc((void **)&d_endoRow, common.endo_mem);
 dpct::dpct_memcpy(d_endoRow, endoRow, common.endo_mem, dpct::host_to_device);

  int* d_endoCol;
 dpct::dpct_malloc((void **)&d_endoCol, common.endo_mem);
 dpct::dpct_memcpy(d_endoCol, endoCol, common.endo_mem, dpct::host_to_device);

  int* d_tEndoRowLoc;
  int* d_tEndoColLoc;
 dpct::dpct_malloc((void **)&d_tEndoRowLoc, common.endo_mem * common.no_frames);
 dpct::dpct_memcpy(d_tEndoRowLoc, tEndoRowLoc,
                   common.endo_mem * common.no_frames, dpct::host_to_device);
 dpct::dpct_malloc((void **)&d_tEndoColLoc, common.endo_mem * common.no_frames);
 dpct::dpct_memcpy(d_tEndoColLoc, tEndoColLoc,
                   common.endo_mem * common.no_frames, dpct::host_to_device);

  int* d_epiRow;
  int* d_epiCol;
 dpct::dpct_malloc((void **)&d_epiRow, common.epi_mem);
 dpct::dpct_memcpy(d_epiRow, epiRow, common.epi_mem, dpct::host_to_device);
 dpct::dpct_malloc((void **)&d_epiCol, common.epi_mem);
 dpct::dpct_memcpy(d_epiCol, epiCol, common.epi_mem, dpct::host_to_device);

  int* d_tEpiRowLoc;
  int* d_tEpiColLoc;
 dpct::dpct_malloc((void **)&d_tEpiRowLoc, common.epi_mem * common.no_frames);
 dpct::dpct_memcpy(d_tEpiRowLoc, tEpiRowLoc, common.epi_mem * common.no_frames,
                   dpct::host_to_device);
 dpct::dpct_malloc((void **)&d_tEpiColLoc, common.epi_mem * common.no_frames);
 dpct::dpct_memcpy(d_tEpiColLoc, tEpiColLoc, common.epi_mem * common.no_frames,
                   dpct::host_to_device);

  //buffer<fp,1> d_mask_conv(common.mask_conv_elem * common.allPoints);
  //d_mask_conv.set_final_data(nullptr);
  fp* d_mask_conv;
 dpct::dpct_malloc((void **)&d_mask_conv,
                   common.mask_conv_mem * common.allPoints);

  //printf("%d\n", common.mask_conv_elem * common.allPoints);
  //buffer<fp,1> d_in_mod_temp(common.in_elem * common.allPoints);
  //d_in_mod_temp.set_final_data(nullptr);
  fp* d_in_mod_temp;
 dpct::dpct_malloc((void **)&d_in_mod_temp, common.in_mem * common.allPoints);

  //printf("%d\n", common.in_elem * common.allPoints);
  //buffer<fp,1> d_in_partial_sum(common.in_cols * common.allPoints);
  //d_in_partial_sum.set_final_data(nullptr);

  fp* d_in_partial_sum;
 dpct::dpct_malloc((void **)&d_in_partial_sum,
                   sizeof(fp) * common.in_cols * common.allPoints);

  //printf("%d\n", common.in_cols * common.allPoints);
  //buffer<fp,1> d_in_sqr_partial_sum(common.in_sqr_rows * common.allPoints);
  //d_in_sqr_partial_sum.set_final_data(nullptr);

  fp* d_in_sqr_partial_sum;
 dpct::dpct_malloc((void **)&d_in_sqr_partial_sum,
                   sizeof(fp) * common.in_sqr_rows * common.allPoints);

  //printf("%d\n", common.in_sqr_rows * common.allPoints);
  //buffer<fp,1> d_par_max_val(common.mask_conv_rows * common.allPoints);
  //d_par_max_val.set_final_data(nullptr);

  fp* d_par_max_val;
 dpct::dpct_malloc((void **)&d_par_max_val,
                   sizeof(fp) * common.mask_conv_rows * common.allPoints);

  //printf("%d\n", common.mask_conv_rows * common.allPoints);
  //buffer<int,1> d_par_max_coo( common.mask_conv_rows * common.allPoints);
  //d_par_max_coo.set_final_data(nullptr);

  fp* d_par_max_coo;
 dpct::dpct_malloc((void **)&d_par_max_coo,
                   sizeof(fp) * common.mask_conv_rows * common.allPoints);

  //buffer<fp,1> d_in_final_sum(common.allPoints);
  //d_in_final_sum.set_final_data(nullptr);

  fp* d_in_final_sum;
 dpct::dpct_malloc((void **)&d_in_final_sum, sizeof(fp) * common.allPoints);

  //buffer<fp,1> d_in_sqr_final_sum(common.allPoints);
  //d_in_sqr_final_sum.set_final_data(nullptr);
  fp* d_in_sqr_final_sum;
 dpct::dpct_malloc((void **)&d_in_sqr_final_sum, sizeof(fp) * common.allPoints);

  //buffer<fp,1> d_denomT(common.allPoints);
  //d_denomT.set_final_data(nullptr);

  fp* d_denomT;
 dpct::dpct_malloc((void **)&d_denomT, sizeof(fp) * common.allPoints);

#ifdef TEST_CHECKSUM
  //buffer<fp,1> d_checksum(CHECK);
  //d_checksum.set_final_data(nullptr);
  //printf("%d\n", CHECK);
  fp* checksum = (fp*) malloc (sizeof(fp)*CHECK);
  fp* d_checksum;
 dpct::dpct_malloc((void **)&d_checksum, sizeof(fp) * CHECK);
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
 dpct::dpct_malloc((void **)&d_frame, sizeof(fp) * common.frame_elem);

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
  dpct::dpct_memcpy(d_frame, frame, sizeof(fp) * common.frame_elem,
                    dpct::host_to_device);

    //==================================================50
    //  launch kernel
    //==================================================50
  {
   dpct::buffer_t d_frame_buf_ct2 = dpct::get_buffer(d_frame);
   dpct::buffer_t d_endoRow_buf_ct3 = dpct::get_buffer(d_endoRow);
   dpct::buffer_t d_endoCol_buf_ct4 = dpct::get_buffer(d_endoCol);
   dpct::buffer_t d_tEndoRowLoc_buf_ct5 = dpct::get_buffer(d_tEndoRowLoc);
   dpct::buffer_t d_tEndoColLoc_buf_ct6 = dpct::get_buffer(d_tEndoColLoc);
   dpct::buffer_t d_epiRow_buf_ct7 = dpct::get_buffer(d_epiRow);
   dpct::buffer_t d_epiCol_buf_ct8 = dpct::get_buffer(d_epiCol);
   dpct::buffer_t d_tEpiRowLoc_buf_ct9 = dpct::get_buffer(d_tEpiRowLoc);
   dpct::buffer_t d_tEpiColLoc_buf_ct10 = dpct::get_buffer(d_tEpiColLoc);
   dpct::buffer_t d_endoT_buf_ct11 = dpct::get_buffer(d_endoT);
   dpct::buffer_t d_epiT_buf_ct12 = dpct::get_buffer(d_epiT);
   dpct::buffer_t d_in2_buf_ct13 = dpct::get_buffer(d_in2);
   dpct::buffer_t d_conv_buf_ct14 = dpct::get_buffer(d_conv);
   dpct::buffer_t d_in2_pad_cumv_buf_ct15 = dpct::get_buffer(d_in2_pad_cumv);
   dpct::buffer_t d_in2_pad_cumv_sel_buf_ct16 =
       dpct::get_buffer(d_in2_pad_cumv_sel);
   dpct::buffer_t d_in2_sub_cumh_buf_ct17 = dpct::get_buffer(d_in2_sub_cumh);
   dpct::buffer_t d_in2_sub_cumh_sel_buf_ct18 =
       dpct::get_buffer(d_in2_sub_cumh_sel);
   dpct::buffer_t d_in2_sub2_buf_ct19 = dpct::get_buffer(d_in2_sub2);
   dpct::buffer_t d_in2_sqr_buf_ct20 = dpct::get_buffer(d_in2_sqr);
   dpct::buffer_t d_in2_sqr_sub2_buf_ct21 = dpct::get_buffer(d_in2_sqr_sub2);
   dpct::buffer_t d_in_sqr_buf_ct22 = dpct::get_buffer(d_in_sqr);
   dpct::buffer_t d_tMask_buf_ct23 = dpct::get_buffer(d_tMask);
   dpct::buffer_t d_mask_conv_buf_ct24 = dpct::get_buffer(d_mask_conv);
   dpct::buffer_t d_in_mod_temp_buf_ct25 = dpct::get_buffer(d_in_mod_temp);
   dpct::buffer_t d_in_partial_sum_buf_ct26 =
       dpct::get_buffer(d_in_partial_sum);
   dpct::buffer_t d_in_sqr_partial_sum_buf_ct27 =
       dpct::get_buffer(d_in_sqr_partial_sum);
   dpct::buffer_t d_par_max_val_buf_ct28 = dpct::get_buffer(d_par_max_val);
   dpct::buffer_t d_par_max_coo_buf_ct29 = dpct::get_buffer(d_par_max_coo);
   dpct::buffer_t d_in_final_sum_buf_ct30 = dpct::get_buffer(d_in_final_sum);
   dpct::buffer_t d_in_sqr_final_sum_buf_ct31 =
       dpct::get_buffer(d_in_sqr_final_sum);
   dpct::buffer_t d_denomT_buf_ct32 = dpct::get_buffer(d_denomT);
#ifdef TEST_CHECKSUM
   dpct::buffer_t d_checksum_buf_ct33 = dpct::get_buffer(d_checksum);
#endif
   dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto d_frame_acc_ct2 =
        d_frame_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
    auto d_endoRow_acc_ct3 =
        d_endoRow_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
    auto d_endoCol_acc_ct4 =
        d_endoCol_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
    auto d_tEndoRowLoc_acc_ct5 =
        d_tEndoRowLoc_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
    auto d_tEndoColLoc_acc_ct6 =
        d_tEndoColLoc_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
    auto d_epiRow_acc_ct7 =
        d_epiRow_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);
    auto d_epiCol_acc_ct8 =
        d_epiCol_buf_ct8.get_access<sycl::access::mode::read_write>(cgh);
    auto d_tEpiRowLoc_acc_ct9 =
        d_tEpiRowLoc_buf_ct9.get_access<sycl::access::mode::read_write>(cgh);
    auto d_tEpiColLoc_acc_ct10 =
        d_tEpiColLoc_buf_ct10.get_access<sycl::access::mode::read_write>(cgh);
    auto d_endoT_acc_ct11 =
        d_endoT_buf_ct11.get_access<sycl::access::mode::read_write>(cgh);
    auto d_epiT_acc_ct12 =
        d_epiT_buf_ct12.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in2_acc_ct13 =
        d_in2_buf_ct13.get_access<sycl::access::mode::read_write>(cgh);
    auto d_conv_acc_ct14 =
        d_conv_buf_ct14.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in2_pad_cumv_acc_ct15 =
        d_in2_pad_cumv_buf_ct15.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in2_pad_cumv_sel_acc_ct16 =
        d_in2_pad_cumv_sel_buf_ct16.get_access<sycl::access::mode::read_write>(
            cgh);
    auto d_in2_sub_cumh_acc_ct17 =
        d_in2_sub_cumh_buf_ct17.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in2_sub_cumh_sel_acc_ct18 =
        d_in2_sub_cumh_sel_buf_ct18.get_access<sycl::access::mode::read_write>(
            cgh);
    auto d_in2_sub2_acc_ct19 =
        d_in2_sub2_buf_ct19.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in2_sqr_acc_ct20 =
        d_in2_sqr_buf_ct20.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in2_sqr_sub2_acc_ct21 =
        d_in2_sqr_sub2_buf_ct21.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in_sqr_acc_ct22 =
        d_in_sqr_buf_ct22.get_access<sycl::access::mode::read_write>(cgh);
    auto d_tMask_acc_ct23 =
        d_tMask_buf_ct23.get_access<sycl::access::mode::read_write>(cgh);
    auto d_mask_conv_acc_ct24 =
        d_mask_conv_buf_ct24.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in_mod_temp_acc_ct25 =
        d_in_mod_temp_buf_ct25.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in_partial_sum_acc_ct26 =
        d_in_partial_sum_buf_ct26.get_access<sycl::access::mode::read_write>(
            cgh);
    auto d_in_sqr_partial_sum_acc_ct27 =
        d_in_sqr_partial_sum_buf_ct27
            .get_access<sycl::access::mode::read_write>(cgh);
    auto d_par_max_val_acc_ct28 =
        d_par_max_val_buf_ct28.get_access<sycl::access::mode::read_write>(cgh);
    auto d_par_max_coo_acc_ct29 =
        d_par_max_coo_buf_ct29.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in_final_sum_acc_ct30 =
        d_in_final_sum_buf_ct30.get_access<sycl::access::mode::read_write>(cgh);
    auto d_in_sqr_final_sum_acc_ct31 =
        d_in_sqr_final_sum_buf_ct31.get_access<sycl::access::mode::read_write>(
            cgh);
    auto d_denomT_acc_ct32 =
        d_denomT_buf_ct32.get_access<sycl::access::mode::read_write>(cgh);
#ifdef TEST_CHECKSUM
    auto d_checksum_acc_ct33 =
        d_checksum_buf_ct33.get_access<sycl::access::mode::read_write>(cgh);
#endif

    auto dpct_global_range = grids * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
         hw(frame_no, common, (float *)(&d_frame_acc_ct2[0]),
            (int *)(&d_endoRow_acc_ct3[0]), (int *)(&d_endoCol_acc_ct4[0]),
            (int *)(&d_tEndoRowLoc_acc_ct5[0]),
            (int *)(&d_tEndoColLoc_acc_ct6[0]), (int *)(&d_epiRow_acc_ct7[0]),
            (int *)(&d_epiCol_acc_ct8[0]), (int *)(&d_tEpiRowLoc_acc_ct9[0]),
            (int *)(&d_tEpiColLoc_acc_ct10[0]), (float *)(&d_endoT_acc_ct11[0]),
            (float *)(&d_epiT_acc_ct12[0]), (float *)(&d_in2_acc_ct13[0]),
            (float *)(&d_conv_acc_ct14[0]),
            (float *)(&d_in2_pad_cumv_acc_ct15[0]),
            (float *)(&d_in2_pad_cumv_sel_acc_ct16[0]),
            (float *)(&d_in2_sub_cumh_acc_ct17[0]),
            (float *)(&d_in2_sub_cumh_sel_acc_ct18[0]),
            (float *)(&d_in2_sub2_acc_ct19[0]),
            (float *)(&d_in2_sqr_acc_ct20[0]),
            (float *)(&d_in2_sqr_sub2_acc_ct21[0]),
            (float *)(&d_in_sqr_acc_ct22[0]), (float *)(&d_tMask_acc_ct23[0]),
            (float *)(&d_mask_conv_acc_ct24[0]),
            (float *)(&d_in_mod_temp_acc_ct25[0]),
            (float *)(&d_in_partial_sum_acc_ct26[0]),
            (float *)(&d_in_sqr_partial_sum_acc_ct27[0]),
            (float *)(&d_par_max_val_acc_ct28[0]),
            (float *)(&d_par_max_coo_acc_ct29[0]),
            (float *)(&d_in_final_sum_acc_ct30[0]),
            (float *)(&d_in_sqr_final_sum_acc_ct31[0]),
            (float *)(&d_denomT_acc_ct32[0]),
#ifdef TEST_CHECKSUM
            (float *)(&d_checksum_acc_ct33[0]), 
#endif
            item_ct1);
        });
   });
  }

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
  dpct::dpct_memcpy(checksum, d_checksum, sizeof(fp) * CHECK,
                    dpct::device_to_host);
    printf("CHECKSUM:\n");
    for(int i=0; i<CHECK; i++){
      printf("i=%d checksum=%f\n", i, checksum[i]);
    }
    printf("\n\n");
#endif

  }

 dpct::dpct_memcpy(tEndoRowLoc, d_tEndoRowLoc,
                   common.endo_mem * common.no_frames, dpct::device_to_host);
 dpct::dpct_memcpy(tEndoColLoc, d_tEndoColLoc,
                   common.endo_mem * common.no_frames, dpct::device_to_host);
 dpct::dpct_memcpy(tEpiRowLoc, d_tEpiRowLoc, common.epi_mem * common.no_frames,
                   dpct::device_to_host);
 dpct::dpct_memcpy(tEpiColLoc, d_tEpiColLoc, common.epi_mem * common.no_frames,
                   dpct::device_to_host);

  //====================================================================================================100
  //  PRINT FRAME PROGRESS END
  //====================================================================================================100
#ifdef TEST_CHECKSUM
  free(checksum);
 dpct::dpct_free(d_checksum);
#endif
 dpct::dpct_free(d_epiT);
 dpct::dpct_free(d_in2);
 dpct::dpct_free(d_conv);
 dpct::dpct_free(d_in2_pad_cumv);
 dpct::dpct_free(d_in2_pad_cumv_sel);
 dpct::dpct_free(d_in2_sub_cumh);
 dpct::dpct_free(d_in2_sub_cumh_sel);
 dpct::dpct_free(d_in2_sub2);
 dpct::dpct_free(d_in2_sqr);
 dpct::dpct_free(d_in2_sqr_sub2);
 dpct::dpct_free(d_in_sqr);
 dpct::dpct_free(d_tMask);
 dpct::dpct_free(d_endoRow);
 dpct::dpct_free(d_endoCol);
 dpct::dpct_free(d_tEndoRowLoc);
 dpct::dpct_free(d_tEndoColLoc);
 dpct::dpct_free(d_epiRow);
 dpct::dpct_free(d_epiCol);
 dpct::dpct_free(d_tEpiRowLoc);
 dpct::dpct_free(d_tEpiColLoc);
 dpct::dpct_free(d_mask_conv);
 dpct::dpct_free(d_in_mod_temp);
 dpct::dpct_free(d_in_partial_sum);
 dpct::dpct_free(d_in_sqr_partial_sum);
 dpct::dpct_free(d_par_max_val);
 dpct::dpct_free(d_par_max_coo);
 dpct::dpct_free(d_in_final_sum);
 dpct::dpct_free(d_in_sqr_final_sum);
 dpct::dpct_free(d_denomT);
 dpct::dpct_free(d_frame);

  printf("\n");
  fflush(NULL);

}

