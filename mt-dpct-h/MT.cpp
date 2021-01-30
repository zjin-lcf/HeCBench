/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

///////////////////////////////////////////////////////////////////////////////
// This sample implements Mersenne Twister random number generator
// and Cartesian Box-Muller transformation on the GPU
///////////////////////////////////////////////////////////////////////////////

// standard utilities and systems includes
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "MT.h"

// comment the below line if not doing Box-Muller transformation
#define DO_BOXMULLER

// Reference CPU MT and Box-Muller transformation 
extern "C" void initMTRef(const char *fname);
extern "C" void RandomRef(float *h_Rand, int nPerRng, unsigned int seed);
#ifdef DO_BOXMULLER
extern "C" void BoxMullerRef(float *h_Rand, int nPerRng);
#endif

using namespace std::chrono;

///////////////////////////////////////////////////////////////////////////////
//Load twister configurations
///////////////////////////////////////////////////////////////////////////////
void loadMTGPU(const char *fname, 
    const unsigned int seed, 
    mt_struct_stripped *h_MT,
    const size_t size)
{
  // open the file for binary read
  FILE* fd = fopen(fname, "rb");
  if (fd == NULL) 
  {
    printf("Failed to open file %s\n", fname);
    exit(-1);
  }

  for (unsigned int i = 0; i < size; i++)
    fread(&h_MT[i], sizeof(mt_struct_stripped), 1, fd);
  fclose(fd);

  for(unsigned int i = 0; i < size; i++)
    h_MT[i].seed = seed;
}



void BoxMullerTrans(float *u1, float *u2)
{
    const float r = sycl::sqrt(-2.0f * sycl::log(*u1));
  const float phi = 2 * PI * (*u2);
    *u1 = r * sycl::cos((float)phi);
    *u2 = r * sycl::sin((float)phi);
}

void boxmuller (float* Rand, const int nPerRng, sycl::nd_item<3> item_ct1) 
{
    int globalID = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                   item_ct1.get_local_id(2);
  for (int iOut = 0; iOut < nPerRng; iOut += 2) {
    BoxMullerTrans(&Rand[globalID + (iOut + 0) * MT_RNG_COUNT],
        &Rand[globalID + (iOut + 1) * MT_RNG_COUNT]);
  }
}

void mt (const mt_struct_stripped* MT, float* Rand, const int nPerRng,
         sycl::nd_item<3> item_ct1) 
{
    int globalID = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                   item_ct1.get_local_id(2);

  int iState, iState1, iStateM, iOut;
  unsigned int mti, mti1, mtiM, x;
  unsigned int mt[MT_NN], matrix_a, mask_b, mask_c; 

  //Load bit-vector Mersenne Twister parameters
  matrix_a = MT[globalID].matrix_a;
  mask_b   = MT[globalID].mask_b;
  mask_c   = MT[globalID].mask_c;

  //Initialize current state
  mt[0] = MT[globalID].seed;
  for (iState = 1; iState < MT_NN; iState++)
    mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

  iState = 0;
  mti1 = mt[0];
  for (iOut = 0; iOut < nPerRng; iOut++) {
    iState1 = iState + 1;
    iStateM = iState + MT_MM;
    if(iState1 >= MT_NN) iState1 -= MT_NN;
    if(iStateM >= MT_NN) iStateM -= MT_NN;
    mti  = mti1;
    mti1 = mt[iState1];
    mtiM = mt[iStateM];

    // MT recurrence
    x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
    x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

    mt[iState] = x;
    iState = iState1;

    //Tempering transformation
    x ^= (x >> MT_SHIFT0);
    x ^= (x << MT_SHIFTB) & mask_b;
    x ^= (x << MT_SHIFTC) & mask_c;
    x ^= (x >> MT_SHIFT1);

    //Convert to (0, 1] float and write to global memory
    Rand[globalID + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Main function 
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  size_t globalWorkSize = {MT_RNG_COUNT};  // 1D var for Total # of work items
  size_t localWorkSize = {128};            // 1D var for # of work items in the work group
    sycl::range<3> gridBlocks(globalWorkSize / localWorkSize, 1, 1);
    sycl::range<3> threadBlocks(localWorkSize, 1, 1);

  const int seed = 777;
  const int nPerRng = 5860;                // # of recurrence steps, must be even if do Box-Muller transformation
  const int nRand = MT_RNG_COUNT * nPerRng;// Output size

  printf("Initialization: load MT parameters and init host buffers...\n");
  mt_struct_stripped *h_MT = (mt_struct_stripped*) malloc (
      sizeof(mt_struct_stripped) * MT_RNG_COUNT); // MT para

  const char *cDatPath = "./data/MersenneTwister.dat";
  loadMTGPU(cDatPath, seed, h_MT, MT_RNG_COUNT);

  const char *cRawPath = "./data/MersenneTwister.raw";
  initMTRef(cRawPath);

  float *h_RandGPU = (float*)malloc(sizeof(float)*nRand); // Host buffers for GPU output
  float *h_RandCPU = (float*)malloc(sizeof(float)*nRand); // Host buffers for CPU test

  printf("Allocate memory...\n"); 

  mt_struct_stripped* d_MT;
  float* d_Rand;
    dpct::dpct_malloc((void **)&d_MT,
                      sizeof(mt_struct_stripped) * MT_RNG_COUNT);
    dpct::dpct_memcpy(d_MT, h_MT, sizeof(mt_struct_stripped) * MT_RNG_COUNT,
                      dpct::host_to_device);
    dpct::dpct_malloc((void **)&d_Rand, sizeof(float) * nRand);

  int numIterations = 100;
  printf("Call Mersenne Twister kernel... (%d iterations)\n\n", numIterations); 
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i = -1; i < numIterations; i++)
  {
        {
            dpct::buffer_t d_MT_buf_ct0 = dpct::get_buffer(d_MT);
            dpct::buffer_t d_Rand_buf_ct1 = dpct::get_buffer(d_Rand);
            q_ct1.submit([&](sycl::handler &cgh) {
                auto d_MT_acc_ct0 =
                    d_MT_buf_ct0.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_Rand_acc_ct1 =
                    d_Rand_buf_ct1.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = gridBlocks * threadBlocks;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadBlocks.get(2),
                                                     threadBlocks.get(1),
                                                     threadBlocks.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        mt((const mt_struct_stripped *)(&d_MT_acc_ct0[0]),
                           (float *)(&d_Rand_acc_ct1[0]), nPerRng, item_ct1);
                    });
            });
        }

#ifdef DO_BOXMULLER
        {
            dpct::buffer_t d_Rand_buf_ct0 = dpct::get_buffer(d_Rand);
            q_ct1.submit([&](sycl::handler &cgh) {
                auto d_Rand_acc_ct0 =
                    d_Rand_buf_ct0.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = gridBlocks * threadBlocks;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadBlocks.get(2),
                                                     threadBlocks.get(1),
                                                     threadBlocks.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        boxmuller((float *)(&d_Rand_acc_ct0[0]), nPerRng,
                                  item_ct1);
                    });
            });
        }
#endif
  }
    dev_ct1.queues_wait_and_throw();

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  double gpuTime = time_span.count() / (double)numIterations;
  printf("MersenneTwister, Throughput = %.4f GNumbers/s, "
      "Time = %.5f s, Size = %u Numbers, Workgroup = %lu\n", 
      ((double)nRand * 1.0E-9 / gpuTime), gpuTime, nRand, localWorkSize);    

  printf("\nRead back results...\n");
    dpct::dpct_memcpy(h_RandGPU, d_Rand, sizeof(float) * nRand,
                      dpct::device_to_host);

  printf("Compute CPU reference solution...\n");
  {
    RandomRef(h_RandCPU, nPerRng, seed);
#ifdef DO_BOXMULLER
    BoxMullerRef(h_RandCPU, nPerRng);
#endif
  }

  printf("Compare CPU and GPU results...\n");
  double sum_delta = 0;
  double sum_ref   = 0;
  {
    for(int i = 0; i < MT_RNG_COUNT; i++)
      for(int j = 0; j < nPerRng; j++) {
        double rCPU = h_RandCPU[i * nPerRng + j];
        double rGPU = h_RandGPU[i + j * MT_RNG_COUNT];
                double delta = std::fabs(rCPU - rGPU);
        sum_delta += delta;
                sum_ref += std::fabs(rCPU);
      }
  }
  double L1norm = sum_delta / sum_ref;
  printf("L1 norm: %E\n\n", L1norm);

  free(h_MT);
  free(h_RandGPU);
  free(h_RandCPU);
    dpct::dpct_free(d_MT);
    dpct::dpct_free(d_Rand);

  // finish
  if (L1norm < 1e-6)
    printf("PASSED\n");
  else
    printf("FAILED\n");

  return 0;
}
