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

// *********************************************************************
//
// Demonstrates host->GPU and GPU->host copies that are asynchronous/overlapped
// with respect to GPU computation (and with respect to host thread).
//
// Because the overlap acheivable for this computation and data set on a given system depends upon the GPU being used and the
// GPU/Host bandwidth, the sample adjust the computation duration to test the most ideal case and test against a consistent standard.
// This sample should be able to achieve up to 30% overlap on GPU's arch 1.2 and 1.3, and up to 50% on arch 2.0+ (Fermi) GPU's.
//
// After setup, warmup and calibration to the system, the sample runs 4 scenarios:
//      A) Computations with 2 command queues on GPU
//         A multiple-cycle sequence is executed, timed and compared against the host
//      B) Computations with 1 command queue on GPU
//         A multiple-cycle sequence is executed, timed and compared against the host
//
//      The 2-command queue approach ought to be substantially faster
//
//      If we name the time to copy single input vector H2D (or outpute vector D2H) as "T", then the optimum comparison case is:
//
//          Single Queue with all the data and all the work
//             Ttot (serial)        = 4T + 4T + 2T      = 10T
//
//          Dual Queue, where each queue has 1/2 the data and 1/2 the work
//             Tq0  (overlap)       = 2T + 2T + T ....
//             Tq1  (overlap)       = .... 2T + 2T + T
//
//             Ttot (elapsed, wall) = 2T + 2T + 2T + T  = 7T
//
//          Best Overlap % = 100.0 * (10T - 7T)/10T = 30.0 %  (Tesla arch 1.2 or 1.3, single copy engine)
//
//      For multiple independent cycles using arch >= 2.0 with 2 copy engines, input and output copies can also be overlapped.
//      This doesn't help for the first cycle, but theoretically can lead to 50% overlap over many independent cycles.
// *********************************************************************

#include <algorithm> // std::min
#include <chrono>
#include <cfloat>    // DBL_MAX
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include "kernel.h"
#include "reference.h"

// GPU error checking macro
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Best possible and Min ratio of compute/copy overlap timing benefit to pass the test
// values greater than 0.0f represent a speed-up relative to non-overlapped
#define EXPECTED_OVERLAP 30.0f
#define PASS_FACTOR 0.60f
#define RETRIES_ON_FAILURE 1

// Base sizes for parameters manipulated dynamically or on the command line
#define BASE_ARRAY_LENGTH 33554432
#define BASE_LOOP_COUNT 32

// the number of stream implies workload partition
// So a single queue allows for four streams. All tasks are just submitted to the single queue
#define nStreams  4

uchar* input = NULL;              // Mapped pointer for pinned Host source A buffer
uchar* output = NULL;             // Mapped pointer for pinned Host result buffer

// demo config vars
uchar* expected_output = NULL;    // temp buffer to hold expected_output results for cross check
size_t szGlobalWorkSize;          // 1D var for Total # of work items in the launched ND range
size_t szLocalWorkSize;           // initial # of work items in the work group

// Forward Declarations
double MultiQueueSequence( uchar *d_input, uchar *d_output,
                           int iCycles, size_t uiNumElements, bool bShowConfig);

double OneQueueSequence( uchar *d_input, uchar *d_output,
                         int iCycles, size_t uiNumElements, bool bShowConfig);

int main(int argc, const char **argv)
{
  const int iWarmupCycles = 4;             // How many times to run the warmup sequence
  uint uiWorkGroup = 256;                  // Command line var workgroup size
  uint uiSize = BASE_ARRAY_LENGTH;         // Command line var (using "sizemult=<n>") to optionally increase vector sizes
  uint uiTestCycle = 100;                  // Command line var (using "sizemult=<n>") to optionally increase vector sizes
  bool bPassFlag = false;                  // Var to accumulate test pass/fail
  int status = false;                      // Cross check result
  bool bTestOverlap = false;
  double dAvgGPUTime[2] = {0.0, 0.0};      // Average time of iTestCycles calls for N-Queue and 1-Queue test
  double dHostTime[2] = {0.0, 0.0};        // Host computation time (2nd test is redundant but a good stability indicator)
  float fMinPassCriteria[2] = {0.0f, 0.0f};// Test pass cireria, adjusted dependant on GPU arch

  uint uiNumElements = uiSize;
  uint iTestCycles = uiTestCycle;

  size_t n = uiSize;
  size_t n2 = uiSize > 1 ? uiSize*2+1 : 5;
  char padCount = n % 3;
  size_t numBlock;
  char padding = (padCount > 0) ? 1 : 0;
  if (padding)
    numBlock = n / 3 + 1; // allocate at most numBlock 3-byte groups
  else
    numBlock = n / 3;
  const size_t streamSize = (numBlock + nStreams - 1) / nStreams;
  const size_t streamBytes = streamSize * 3; // each block has three bytes
  const size_t  bytes = streamBytes * nStreams; // n;
  const size_t bytes2 = n2;

  printf("Input array sizes (without padding) = %zu input elements\n", n);
  printf("Output array sizes = %zu output elements\n", n2);

  szLocalWorkSize = uiWorkGroup;

  // Device and Platform eligible for overlap testing
  bTestOverlap = true;

  // If device has overlap capability, proceed
  fMinPassCriteria[0] = PASS_FACTOR * EXPECTED_OVERLAP;               // 1st cycle overlap is same for 1 or 2 copy engines

  // Allocate pinned source and result host buffers:
  //   Note: Pinned (Page Locked) memory is needed for async host<->GPU memory copy operations ***
  GPU_CHECK(cudaMallocHost((void**)&input, bytes));
  GPU_CHECK(cudaMallocHost((void**)&output, bytes2));

  // allocate device buffers
  uchar *d_input;
  GPU_CHECK(cudaMalloc((void**)&d_input, bytes));
  uchar *d_output;
  GPU_CHECK(cudaMalloc((void**)&d_output, bytes2));

  printf("CreateBuffer (Input and Result GPU Device, size in byte: %zu %zu)...\n\n", bytes, bytes2);

  // Alloc expected_output buffer for cross checks
  expected_output = (uchar*)malloc(bytes2);

  size_t encSize;

  printf("Warmup with %d-Queue sequence, %d cycles...\n", nStreams, iWarmupCycles);
  MultiQueueSequence(d_input, d_output, iWarmupCycles, uiNumElements, false);

  for( int iRun =0; iRun < RETRIES_ON_FAILURE; ++iRun ) {

    printf("*******************************************\n");
    printf("Run and time with multiple command-queues\n");
    printf("*******************************************\n");
    // Run the sequence iTestCycles times
    dAvgGPUTime[0] = MultiQueueSequence(d_input, d_output, iTestCycles, uiNumElements, false);

    encSize = 0;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1; i++)
    {
      reference(input, n, expected_output, n2, &encSize);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dHostTime[0] = time * 1e-9;

    // Compare host and GPU results
    printf("  Device vs Host Result Comparison\t: ");
    status = memcmp(expected_output, output, encSize);
    printf("%s\n", status ? "FAIL" : "PASS");
    bPassFlag = (status == 0);

    printf("*******************************************\n");
    printf("Run and time with 1 command queue\n");
    printf("*******************************************\n");
    // Run the sequence iTestCycles times
    dAvgGPUTime[1] = OneQueueSequence(d_input, d_output, iTestCycles, uiNumElements, false);

    encSize = 0;
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1; i++)
    {
      reference(input, n, expected_output, n2, &encSize);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dHostTime[1] = time * 1e-9;

    // Compare host and GPU results
    printf("  Device vs Host Result Comparison\t: ");
    status = memcmp(expected_output, output, encSize);
    printf("%s\n", status ? "FAIL" : "PASS");
    bPassFlag &= (status == 0);

    // Compare Single and Dual queue timing
    printf("\nResult Summary:\n");

    // Log GPU and CPU Time for N-queue scenario
    for (int i = 0; i < 2; i++) {
      printf("  Min GPU Elapsed Time for %d-Queue execution = %.5f s\n",
             i == 0 ? nStreams : 1, dAvgGPUTime[i]);
      printf("  Max GPU Kernel Throughput for %d-Queue execution = %.2f GB/s\n",
             i == 0 ? nStreams : 1, numBlock * 7 / dAvgGPUTime[i] / 1e9);
      printf("  Avg Host Elapsed Time\t\t\t= %.5f s\n\n", dHostTime[i]);
    }

    // Log overlap % for GPU (comparison of 2-queue and 1 queue scenarios) and status
    double dAvgOverlap = 100.0 * (1.0 - dAvgGPUTime[0]/dAvgGPUTime[1]);

    if( bTestOverlap ) {
      bool bAvgOverlapOK = (dAvgOverlap >= fMinPassCriteria[1]);
      if( iRun == RETRIES_ON_FAILURE || bAvgOverlapOK ) {
        printf("  Measured and (Acceptable) Avg Overlap\t= %.1f %% (%.1f %%)  -> Measured Overlap is %s\n\n",
               dAvgOverlap, fMinPassCriteria[1], bAvgOverlapOK ? "Acceptable" : "NOT Acceptable");

        // Log info to master log in standard format
        printf("ComputeOverlap-Avg, Throughput = %.4f OverlapPercent, Time = %.5f s, Size = %u Elements, Workgroup = %lu\n",
               dAvgOverlap, dAvgGPUTime[0], uiNumElements, szLocalWorkSize);

        bPassFlag &= bAvgOverlapOK;
        break;
      }
    }

    printf("  Measured and (Acceptable) Avg Overlap\t= %.1f %% (%.1f %%)  -> Retry %d more time(s)...\n\n",
      dAvgOverlap, fMinPassCriteria[1], RETRIES_ON_FAILURE - iRun);
  }

  printf("Starting Cleanup...\n\n");
  if(expected_output) free(expected_output);
  GPU_CHECK(cudaFreeHost(input));
  GPU_CHECK(cudaFreeHost(output));
  GPU_CHECK(cudaFree(d_input));
  GPU_CHECK(cudaFree(d_output));

  return 0;
}

// Run 1 queue sequence for n cycles
double OneQueueSequence( uchar *d_input, uchar *d_output,
                         int iCycles, size_t numElements, bool bShowConfig)
{
  // Use fresh source Data: (re)initialize pinned host array buffers (using mapped standard pointer to pinned host cl_mem buffer)
  for (size_t i = 0; i < numElements; i++)
    input[i] = rand()%256;  // initialize for both sw/hw run

  // Reset Global work size for 1 command-queue, and log work sizes & dimensions
  char padCount = numElements % 3;
  size_t numBlock;
  char padding = (padCount > 0) ? 1 : 0;
  if (padding)
    numBlock = numElements / 3 + 1; // allocate at most numBlock 3-byte groups
  else
    numBlock = numElements / 3;

  // scheme1
  szLocalWorkSize = std::min(szLocalWorkSize, numBlock);
  szGlobalWorkSize = (numBlock + szLocalWorkSize-1) / szLocalWorkSize;

  double dAvgTime = DBL_MAX; //0.0;

  printf("Run the sequence %d times\n", iCycles);
  for (int i = 0; i < iCycles; i++)
  {
    auto start = std::chrono::steady_clock::now();
    // Nonblocking Write of all of input data from host to device in command-queue 0
    GPU_CHECK(cudaMemcpy(d_input, input, numBlock * 3, cudaMemcpyHostToDevice));

    // Launch kernel computation
    base64_enc<<<szGlobalWorkSize, szLocalWorkSize>>>(d_input, d_output, padCount, numBlock, 0);

    // Non Blocking Read of output data from device to host, command-queue 0
    GPU_CHECK(cudaMemcpy(output, d_output, numBlock * 4, cudaMemcpyDeviceToHost));

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dAvgTime = std::min(dAvgTime, time * 1e-9);
  }

  // Log config if asked for
  if (bShowConfig)
  {
    printf("\n1-Queue sequence Configuration:\n");
  }
  return dAvgTime;
}

// Run 2 queue sequence for n cycles
double MultiQueueSequence( uchar *d_input, uchar *d_output,
                           int iCycles, size_t numElements, bool bShowConfig)
{
  // Locals
  size_t numBlock;
  char padCount = numElements % 3;
  char padding = (padCount > 0) ? 1 : 0;
  if (padding)
    numBlock = numElements / 3 + 1; // allocate at most numBlock 3-byte groups
  else
    numBlock = numElements / 3;
  double dAvgTime = DBL_MAX; //0.0;

  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; i++)
    GPU_CHECK(cudaStreamCreate(&stream[i]));

  // Use fresh source Data: (re)initialize pinned host array buffers (using mapped standard pointer to pinned host cl_mem buffer)
  for (size_t i = 0; i < numElements; i++)
    input[i] = rand()%256;

  // Set Global work size for multiple command-queues, and log work sizes & dimensions
  const size_t streamSize = (numBlock + nStreams - 1) / nStreams;

  szLocalWorkSize = std::min(szLocalWorkSize, streamSize);
  szGlobalWorkSize = (streamSize + szLocalWorkSize-1) / szLocalWorkSize;
  const size_t streamBytes = streamSize * 3; // each block has three bytes
  const size_t streamBytes2 = streamSize * 4; // each block has four bytes

  printf("Number of blocks = %lu\n", numBlock);
  printf("Global Size = %lu\n\n", szGlobalWorkSize);
  printf("Workgroup Size = %lu\n\n", szLocalWorkSize);
  printf("There are %d streams\n", nStreams);
  printf("Input stream size = %lu  (%lu bytes)\n", streamSize, streamBytes);
  printf("Outputput stream size = %lu  (%lu bytes)\n", streamSize, streamBytes2);

  printf("Run the sequence %d times\n", iCycles);
  for (int c = 0; c < iCycles; c++)
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nStreams; i++) {
      size_t input_offset = i * streamBytes;
      size_t output_offset = i * streamBytes2;

      // Nonblocking Write of a chunk of input data from host to device
      GPU_CHECK(cudaMemcpyAsync(d_input+input_offset, input+input_offset, streamBytes, cudaMemcpyHostToDevice, stream[i]));

      base64_enc<<<szGlobalWorkSize, szLocalWorkSize, 0, stream[i]>>> (d_input, d_output, padCount, numBlock, i*streamSize);

      // Non Blocking Read of a chunk of output data from device to host
      GPU_CHECK(cudaMemcpyAsync(output+output_offset, d_output+output_offset, streamBytes2, cudaMemcpyDeviceToHost, stream[i]));
    }

    // Sync to host and get average sequence time
    for (int i = 0; i < nStreams; i++)
      GPU_CHECK(cudaStreamSynchronize(stream[i]));

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dAvgTime = std::min(dAvgTime, time * 1e-9);
  }

  // Log config if asked for
  if (bShowConfig)
  {
    printf("\n%d-Queue sequence Configuration:\n", nStreams);
    printf("  Global Work Size (per command-queue)\t= %lu\n  Local Work Size \t\t\t= %lu\n  # of Work Groups (per command-queue)\t= %lu\n  # of command-queues\t\t\t= 2\n",
        szGlobalWorkSize, szLocalWorkSize, szGlobalWorkSize/szLocalWorkSize);
  }

  for (int i = 0; i < nStreams; i++)
    GPU_CHECK(cudaStreamDestroy(stream[i]));

  return dAvgTime;
}
