/*  Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
  All rights reserved. https://github.com/robertwgh/cuLDPC

  CUDA implementation of LDPC decoding algorithm.

  The details of implementation can be found from the following papers:
  1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
  2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.

  The current release is close to the GlobalSIP2013 paper. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

#include "common.h"
#include "LDPC.h"
#include "matrix.h"

#include "kernel.cpp"


float sigma ;
int *info_bin ;

int main()
{
  printf("GPU LDPC Decoder\r\nComputing...\r\n");

// For cnp kernel
#if MODE == WIMAX
  const char h_element_count1[BLK_ROW] = {6, 7, 7, 6, 6, 7, 6, 6, 7, 6, 6, 6};
  const char h_element_count2[BLK_COL] = {3, 3, 6, 3, 3, 6, 3, 6, 3, 6, 3, 6, \
                                          3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
#else
  const char h_element_count1[BLK_ROW] = {7, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 8};
  const char h_element_count2[BLK_COL] = {11,4, 3, 3,11, 3, 3, 3,11, 3, 3, 3, \
                                          3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
#endif

  h_element h_compact1 [H_COMPACT1_COL*H_COMPACT1_ROW]; // for update dt, R
  h_element h_element_temp;

  // init the compact matrix
  for(int i = 0; i < H_COMPACT1_COL; i++)
  {
    for(int j = 0; j < H_COMPACT1_ROW; j ++)
    {
      h_element_temp.x = 0;
      h_element_temp.y = 0;
      h_element_temp.value = -1;
      h_element_temp.valid = 0;
      h_compact1[i*H_COMPACT1_ROW+j] = h_element_temp; // h[i][0-11], the same column
    }
  }

  // scan the h matrix, and gengerate compact mode of h
  for(int i = 0; i < BLK_ROW; i++)
  {
    int k = 0;
    for(int j = 0; j <  BLK_COL; j ++)
    {
      if(h_base[i][j] != -1)
      {
        h_element_temp.x = i;
        h_element_temp.y = j;
        h_element_temp.value = h_base[i][j];
        h_element_temp.valid = 1;
        h_compact1[k*H_COMPACT1_ROW+i] = h_element_temp; // h[i][0-11], the same column
        k++;
      }
    }
    // printf("row %d, #element=%d\n", i, k);
  }

  // h_compact2
  h_element h_compact2 [H_COMPACT2_ROW*H_COMPACT2_COL]; // for update llr

  // init the compact matrix
  for(int i = 0; i < H_COMPACT2_ROW; i++)
  {
    for(int j = 0; j < H_COMPACT2_COL; j ++)
    {
      h_element_temp.x = 0;
      h_element_temp.y = 0;
      h_element_temp.value = -1;
      h_element_temp.valid = 0;
      h_compact2[i*H_COMPACT2_COL+j] = h_element_temp;
    }
  }

  for(int j = 0; j < BLK_COL; j++)
  {
    int k = 0;
    for(int i = 0; i < BLK_ROW; i ++)
    {
      if(h_base[i][j] != -1)
      {
        // although h is transposed, the (x,y) is still (iBlkRow, iBlkCol)
        h_element_temp.x = i; 
        h_element_temp.y = j;
        h_element_temp.value = h_base[i][j];
        h_element_temp.valid = 1;
        h_compact2[k*H_COMPACT2_COL+j] = h_element_temp;
        k++;
      }
    }
  }

  //int memorySize_h_base = BLK_ROW * BLK_COL * sizeof(int);
  int wordSize_h_compact1 = H_COMPACT1_ROW * H_COMPACT1_COL;
  int wordSize_h_compact2 = H_COMPACT2_ROW * H_COMPACT2_COL;

  int memorySize_infobits = INFO_LEN * sizeof(int);
  int memorySize_codeword = CODEWORD_LEN * sizeof(int);
  int memorySize_llr = CODEWORD_LEN * sizeof(float);

  info_bin = (int *) malloc(memorySize_infobits) ;
  int *codeword = (int *) malloc(memorySize_codeword) ;
  float *trans = (float *) malloc(memorySize_llr) ;
  float *recv = (float *) malloc(memorySize_llr) ;
  float *llr = (float *) malloc(memorySize_llr) ;

  float rate = (float)0.5f;

  //////////////////////////////////////////////////////////////////////////////////
  // all the variables Starting with _gpu is used in host code and for cuda computation
  int wordSize_llr = MCW *  CW * CODEWORD_LEN;
  int wordSize_dt = MCW *  CW * ROW * BLK_COL;
  int wordSize_R = MCW *  CW * ROW * BLK_COL;
  int wordSize_hard_decision = MCW * CW * CODEWORD_LEN;

  int memorySize_infobits_gpu = MCW * CW * memorySize_infobits ;
  int memorySize_llr_gpu = wordSize_llr * sizeof(float);
  //int memorySize_dt_gpu = wordSize_dt * sizeof(float);
  //int memorySize_R_gpu = wordSize_R * sizeof(float);
  int memorySize_hard_decision_gpu = wordSize_hard_decision * sizeof(int);

  int *info_bin_gpu;
  float *llr_gpu;
  int * hard_decision_gpu;

  info_bin_gpu = (int *) malloc(memorySize_infobits_gpu);
  hard_decision_gpu = (int *) malloc(memorySize_hard_decision_gpu);
  llr_gpu = (float *) malloc(memorySize_llr_gpu);

  error_result this_error;

  int total_frame_error = 0;
  int total_bit_error = 0;
  int total_codeword = 0;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  
  buffer<float, 1> d_llr(wordSize_llr);
  buffer<float, 1> d_dt(wordSize_dt);
  buffer<float, 1> d_R(wordSize_R);
  buffer<int, 1> d_hard_decision(wordSize_hard_decision);
  buffer<h_element, 1> d_h_compact1(h_compact1, wordSize_h_compact1);
  buffer<h_element, 1> d_h_compact2(h_compact2, wordSize_h_compact2);
  buffer<char, 1> d_h_element_count1(h_element_count1, BLK_ROW);
  buffer<char, 1> d_h_element_count2(h_element_count2, BLK_COL);

  srand(69012);

  for(int snri = 0; snri < NUM_SNR; snri++)
  {
    float snr = snr_array[snri];
    sigma = 1.0f/sqrt(2.0f*rate*pow(10.0f,(snr/10.0f)));

    total_codeword = 0;
    total_frame_error = 0;
    total_bit_error = 0;

    // Adjust MIN_CODWORD in LDPC.h to reduce simulation time
    while ( (total_frame_error <= MIN_FER) && (total_codeword <= MIN_CODEWORD))
    {
      total_codeword += CW * MCW;

      for(int i = 0; i < CW * MCW; i++)
      {
        // generate random data
        info_gen (info_bin, rand());

        // encode the data
        structure_encode (info_bin, codeword, h_base);

        // BPSK modulation
        modulation (codeword, trans);

        // additive white Gaussian noise
        awgn (trans, recv, rand());

        // LLR init
        llr_init (llr, recv);

        // copy the info_bin and llr to the total memory
        memcpy(info_bin_gpu + i * INFO_LEN, info_bin, memorySize_infobits);
        memcpy(llr_gpu + i * CODEWORD_LEN, llr, memorySize_llr);
      }

      // Define CUDA kernel dimension
      range<2> gws(MCW * CW, BLK_ROW * BLOCK_SIZE_X); // dim of the thread blocks
      range<2> lws(CW, BLOCK_SIZE_X);
      int sharedRCacheSize = THREADS_PER_BLOCK * NON_EMPTY_ELMENT;

      range<2> gws2(MCW * CW, BLK_COL * BLOCK_SIZE_X);
      range<2> lws2(CW, BLOCK_SIZE_X);
      //int sharedDtCacheSize = THREADS_PER_BLOCK * NON_EMPTY_ELMENT_VNP * sizeof(float);

      // run the kernel
      for(int j = 0; j < MAX_SIM; j++)
      {
        // Transfer LLR data into device.
        q.submit([&] (handler &cgh) {
          auto acc = d_llr.get_access<sycl_discard_write>(cgh);
          cgh.copy(llr_gpu, acc);
        });

        // kernel launch
        for(int ii = 0; ii < MAX_ITERATION; ii++)
        {

          // run check-node processing kernel
          // TODO: run a special kernel the first iteration?
          if(ii == 0) {
            q.submit([&] (handler &cgh) {
              auto dev_llr = d_llr.get_access<sycl_read>(cgh);
              auto dev_dt = d_dt.get_access<sycl_discard_write>(cgh);
              auto dev_R = d_R.get_access<sycl_discard_write>(cgh);
              auto dev_h_element_count1 = d_h_element_count1.get_access<sycl_read>(cgh);
              auto dev_h_compact1 = d_h_compact1.get_access<sycl_read>(cgh);
              cgh.parallel_for<class cnp_kernel>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
                ldpc_cnp_kernel_1st_iter (
                  dev_llr.get_pointer(), 
                  dev_dt.get_pointer(), 
                  dev_R.get_pointer(), 
                  dev_h_element_count1.get_pointer(), 
                  dev_h_compact1.get_pointer(),
                  item);
              });
            });
          } else {
            q.submit([&] (handler &cgh) {
              auto dev_llr = d_llr.get_access<sycl_read>(cgh);
              auto dev_dt = d_dt.get_access<sycl_discard_write>(cgh);
              auto dev_R = d_R.get_access<sycl_read_write>(cgh);
              auto dev_h_element_count1 = d_h_element_count1.get_access<sycl_read>(cgh);
              auto dev_h_compact1 = d_h_compact1.get_access<sycl_read>(cgh);
              accessor<float, 1, sycl_read_write, access::target::local> RCache(sharedRCacheSize, cgh);
              cgh.parallel_for<class cnp_kernel2>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
                ldpc_cnp_kernel(
                  dev_llr.get_pointer(),
                  dev_dt.get_pointer(),
                  dev_R.get_pointer(),
                  dev_h_element_count1.get_pointer(), 
                  dev_h_compact1.get_pointer(),
                  RCache.get_pointer(),
                  item);
              });
            });
          }
            //auto h_dt = d_dt.get_access<sycl_read>();
            //auto h_R = d_R.get_access<sycl_read>();
            //for (int i = 0; i < wordSize_dt; i++) printf("%d dt: %f\n", ii, h_dt[i]);
            //for (int i = 0; i < wordSize_R; i++) printf("%d R: %f\n",  ii,h_R[i]);

          // run variable-node processing kernel
          // for the last iteration we run a special
          // kernel. this is because we can make a hard
          // decision instead of writing back the belief
          // for the value of each bit.
          if(ii < MAX_ITERATION - 1) {
            q.submit([&] (handler &cgh) {
              auto dev_llr = d_llr.get_access<sycl_read_write>(cgh);
              auto dev_dt = d_dt.get_access<sycl_read>(cgh);
              auto dev_h_element_count2 = d_h_element_count2.get_access<sycl_read>(cgh);
              auto dev_h_compact2 = d_h_compact2.get_access<sycl_read>(cgh);
              cgh.parallel_for<class vnp_kernel>(nd_range<2>(gws2, lws2), [=] (nd_item<2> item) {
                ldpc_vnp_kernel_normal(
                   dev_llr.get_pointer(), 
                   dev_dt.get_pointer(), 
                   dev_h_element_count2.get_pointer(), 
                   dev_h_compact2.get_pointer(),
                   item);
              });
            });
          } else {
            q.submit([&] (handler &cgh) {
              auto dev_llr = d_llr.get_access<sycl_read>(cgh);
              auto dev_dt = d_dt.get_access<sycl_read>(cgh);
              auto dev_hd = d_hard_decision.get_access<sycl_discard_write>(cgh);
              auto dev_h_element_count2 = d_h_element_count2.get_access<sycl_read>(cgh);
              auto dev_h_compact2 = d_h_compact2.get_access<sycl_read>(cgh);
              cgh.parallel_for<class vnp_kernel2>(nd_range<2>(gws2, lws2), [=] (nd_item<2> item) {
                ldpc_vnp_kernel_last_iter
                (dev_llr.get_pointer(), 
                 dev_dt.get_pointer(), 
                 dev_hd.get_pointer(), 
                 dev_h_element_count2.get_pointer(), 
                 dev_h_compact2.get_pointer(),
                 item);
              });
            });
          }
        }

        // copy the decoded data from device to host
        q.submit([&] (handler &cgh) {
          auto acc = d_hard_decision.get_access<sycl_read>(cgh);
          cgh.copy(acc, hard_decision_gpu); 
        });

        q.wait();

        this_error = cuda_error_check(info_bin_gpu, hard_decision_gpu);
        total_bit_error += this_error.bit_error;
        total_frame_error += this_error.frame_error;
      } // end of MAX-SIM

      printf ("# codewords = %d, CW=%d, MCW=%d\n",total_codeword, CW, MCW);
      printf ("total bit error = %d\n", total_bit_error);
      printf ("total frame error = %d\n", total_frame_error);
      printf ("BER = %1.2e, FER = %1.2e\n", 
          (float) total_bit_error/total_codeword/INFO_LEN, 
          (float) total_frame_error/total_codeword);
    } // end of the MAX frame error.
  }// end of the snr loop

  free(info_bin);
  free(codeword);
  free(trans);
  free(recv);
  free(llr);
  free(llr_gpu);
  free(hard_decision_gpu);
  free(info_bin_gpu);

  return 0;
}
