/*  Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
    All rights reserved. https://github.com/robertwgh/cuLDPC

    CUDA implementation of LDPC decoding algorithm.

    The details of implementation can be found from the following papers:
    1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
    2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.

    The current release is close to the GlobalSIP2013 paper.

Created:   10/1/2010
Revision:  08/01/2013
04/20/2016 prepare for release on Github.
11/26/2017 cleanup and comments by Albin Severinson, albin (at) severinson.org
 */


#include "LDPC.h"


// Kernel 1
void ldpc_cnp_kernel_1st_iter(
    const float * dev_llr,
    float * dev_dt,
    float * dev_R,
    const char * dev_h_element_count1,
    const h_element * dev_h_compact1,
    sycl::nd_item<2> &item)
{
#if MODE == WIFI
  if(item.get_local_id(1) >= Z)
    return;
#endif

  int iCW = item.get_local_id(0); // index of CW in a MCW
  int iMCW = item.get_group(0); // index of MCW
  int iCurrentCW = iMCW * CW + iCW;


  //for step 1: update dt
  int iBlkRow; // block row in h_base
  int iBlkCol; // block col in h_base
  int iSubRow; // row index in sub_block of h_base
  int iCol; // overall col index in h_base
  int offsetR;

  iSubRow = item.get_local_id(1);
  iBlkRow = item.get_group(1);

  int size_llr_CW = COL; // size of one llr CW block
  int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block
  int shift_t;

  // For 2-min algorithm.
  char Q_sign = 0;
  char sq;
  float Q, Q_abs;
  float R_temp;

  float sign = 1.0f;
  float rmin1 = 1000.0f;
  float rmin2 = 1000.0f;
  char idx_min = 0;

  h_element h_element_t;
  int s = dev_h_element_count1[iBlkRow];
  offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

  // The 1st recursion
  for(int i = 0; i < s; i++) // loop through all the ZxZ sub-blocks in a row
  {
    h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];

    iBlkCol = h_element_t.y;
    shift_t = h_element_t.value;

    shift_t = (iSubRow + shift_t);
    if(shift_t >= Z) shift_t = shift_t - Z;

    iCol = iBlkCol * Z + shift_t;

    Q = dev_llr[size_llr_CW * iCurrentCW + iCol];// - R_temp;
    Q_abs = sycl::fabs(Q);
    sq = Q < 0;

    // quick version
    sign = sign * (1 - sq * 2);
    Q_sign |= sq << i;

    if (Q_abs < rmin1)
    {
      rmin2 = rmin1;
      rmin1 = Q_abs;
      idx_min = i;
    } else if (Q_abs < rmin2)
    {
      rmin2 = Q_abs;
    }
  }

  // The 2nd recursion
  for(int i = 0; i < s; i ++)
  {
    // v0: Best performance so far. 0.75f is the value of alpha.
    sq = 1 - 2 * ((Q_sign >> i) & 0x01);
    R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

    // write results to global memory
    h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];
    int addr_temp = offsetR + h_element_t.y * ROW;
    dev_dt[addr_temp] = R_temp;// - R1[i]; // compute the dt value for current llr.
    dev_R[addr_temp] = R_temp; // update R, R=R'.
  }
}

// Kernel_1
void ldpc_cnp_kernel(
    const float * dev_llr,
    float * dev_dt,
    float * dev_R,
    const char * dev_h_element_count1,
    const h_element * dev_h_compact1,
    float * RCache,
    sycl::nd_item<2> &item)
{
#if MODE == WIFI
  if(item.get_local_id(1) >= Z)
    return;
#endif

  // Define cache for R: Rcache[NON_EMPTY_ELMENT][nThreadPerBlock]
  // extern means that the memory is allocated dynamically at run-time
  int iRCacheLine = item.get_local_id(0) * item.get_local_range(1) + item.get_local_id(1);

  int iCW = item.get_local_id(0); // index of CW in a MCW
  int iMCW = item.get_group(0); // index of MCW
  int iCurrentCW = iMCW * CW + iCW;


  //for step 1: update dt
  int iBlkRow; // block row in h_base
  int iBlkCol; // block col in h_base
  int iSubRow; // row index in sub_block of h_base
  int iCol; // overall col index in h_base
  int offsetR;

  iSubRow = item.get_local_id(1);
  iBlkRow = item.get_group(1);

  int size_llr_CW = COL; // size of one llr CW block
  int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

  //float R1[NON_EMPTY_ELMENT];
  int shift_t;

  // For 2-min algorithm.
  char Q_sign = 0;
  char sq;
  float Q, Q_abs;
  float R_temp;

  float sign = 1.0f;
  float rmin1 = 1000.0f;
  float rmin2 = 1000.0f;
  char idx_min = 0;

  h_element h_element_t;
  int s = dev_h_element_count1[iBlkRow];
  offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

  // The 1st recursion
  // TODO: Is s always the same? If so we can unroll the loop with #pragma unroll
  for(int i = 0; i < s; i++) // loop through all the ZxZ sub-blocks in a row
  {
    h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];

    iBlkCol = h_element_t.y;
    shift_t = h_element_t.value;

    shift_t = (iSubRow + shift_t);
    if(shift_t >= Z) shift_t = shift_t - Z;

    iCol = iBlkCol * Z + shift_t;

    R_temp = dev_R[offsetR + iBlkCol * ROW];

    RCache[i * THREADS_PER_BLOCK + iRCacheLine] =  R_temp;

    Q = dev_llr[size_llr_CW * iCurrentCW + iCol] - R_temp;
    Q_abs = sycl::fabs(Q);

    sq = Q < 0;
    sign = sign * (1 - sq * 2);
    Q_sign |= sq << i;

    if (Q_abs < rmin1)
    {
      rmin2 = rmin1;
      rmin1 = Q_abs;
      idx_min = i;
    } else if (Q_abs < rmin2)
    {
      rmin2 = Q_abs;
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  // The 2nd recursion
  //#pragma unroll
  for(int i = 0; i < s; i ++)
  {
    sq = 1 - 2 * ((Q_sign >> i) & 0x01);
    R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

    // write results to global memory
    h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];
    int addr_temp = h_element_t.y * ROW + offsetR;
    dev_dt[addr_temp] = R_temp - RCache[i * THREADS_PER_BLOCK + iRCacheLine];
    dev_R[addr_temp] = R_temp; // update R, R=R'.
  }
}

// Kernel 2: VNP processing
void ldpc_vnp_kernel_normal(
    float * dev_llr,
    const float * dev_dt,
    const char *dev_h_element_count2,
    const h_element *dev_h_compact2,
    sycl::nd_item<2> &item)
{

#if MODE == WIFI
  if(item.get_local_id(1) >= Z)
    return;
#endif

  int  iCW = item.get_local_id(0); // index of CW in a MCW
  int iMCW = item.get_group(0); // index of MCW
  int iCurrentCW = iMCW * CW + iCW;


  int iBlkCol;
  int iBlkRow;
  int iSubCol;
  int iRow;
  int iCol;

  int shift_t, sf;
  int llr_index;
  float APP;

  h_element h_element_t;

  iBlkCol = item.get_group(1);
  iSubCol = item.get_local_id(1);

  int size_llr_CW = COL; // size of one llr CW block
  int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

  // update all the llr values
  iCol = iBlkCol * Z + iSubCol;
  llr_index = size_llr_CW * iCurrentCW + iCol;

  APP = dev_llr[llr_index];
  int offsetDt = size_R_CW * iCurrentCW + iBlkCol * ROW;

  for(int i = 0; i < dev_h_element_count2[iBlkCol]; i++)
  {
    h_element_t = dev_h_compact2[i * H_COMPACT2_COL + iBlkCol];

    shift_t = h_element_t.value;
    iBlkRow = h_element_t.x;

    sf = iSubCol - shift_t;
    if(sf < 0) sf = sf + Z;

    iRow = iBlkRow * Z + sf;
    APP = APP + dev_dt[offsetDt + iRow];
  }
  // write back to device global memory
  dev_llr[llr_index] = APP;

  // No hard decision for non-last iteration.
}

// Kernel: VNP processing for the last iteration.
void ldpc_vnp_kernel_last_iter(
    const float * dev_llr,
    const float * dev_dt,
    int * dev_hd,
    const char *dev_h_element_count2,
    const h_element *dev_h_compact2,
    sycl::nd_item<2> &item)
{
#if MODE == WIFI
  if(item.get_local_id(1) >= Z)
    return;
#endif

  int  iCW = item.get_local_id(0); // index of CW in a MCW
  int iMCW = item.get_group(0); // index of MCW
  int iCurrentCW = iMCW * CW + iCW;


  int iBlkCol;
  int iBlkRow;
  int iSubCol;
  int iRow;
  int iCol;

  int shift_t, sf;
  int llr_index;
  float APP;

  h_element h_element_t;

  iBlkCol = item.get_group(1);
  iSubCol = item.get_local_id(1);

  int size_llr_CW = COL; // size of one llr CW block
  int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

  // update all the llr values
  iCol = iBlkCol * Z + iSubCol;
  llr_index = size_llr_CW * iCurrentCW + iCol;

  APP = dev_llr[llr_index];

  int offsetDt = size_R_CW * iCurrentCW + iBlkCol * ROW;

  for(int i = 0; i < dev_h_element_count2[iBlkCol]; i ++)
  {
    h_element_t = dev_h_compact2[i * H_COMPACT2_COL + iBlkCol];

    shift_t = h_element_t.value;
    iBlkRow = h_element_t.x;

    sf = iSubCol - shift_t;
    if(sf < 0) sf = sf + Z;

    iRow = iBlkRow * Z + sf;
    APP = APP + dev_dt[offsetDt + iRow];
  }

  // For the last iteration, we don't need to write intermediate results to
  // global memory. Instead, we directly make a hard decision.
  if(APP > 0)
    dev_hd[llr_index] = 0;
  else
    dev_hd[llr_index] = 1;
}
