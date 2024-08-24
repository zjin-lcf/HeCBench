#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cmath>
#include <chrono>
#include "utils.h"

#define BLOCK_SIZE 2048    // in unit of byte, the size of one data block
#define THREAD_SIZE 128     // in unit of datatype, the size of the thread block, so as the size of symbols per iteration
#define WINDOW_SIZE 32     // in unit of datatype, maximum 255, the size of the sliding window, so as the maximum match length
#define INPUT_TYPE uint32_t // define input type

// Define the compress match kernel functions
template <typename T>
void compressKernelI(const T *input, uint32_t numOfBlocks,
                     uint32_t *__restrict flagArrSizeGlobal,
                     uint32_t *__restrict compressedDataSizeGlobal,
                     uint8_t *__restrict tmpFlagArrGlobal,
                     uint8_t *__restrict tmpCompressedDataGlobal,
                     int minEncodeLength,
                     const sycl::nd_item<3> &item,
                     T *__restrict buffer,
                     uint8_t *__restrict lengthBuffer,
                     uint8_t *__restrict offsetBuffer,
                     uint32_t *__restrict prefixBuffer,
                     uint8_t *__restrict byteFlagArr)
{
  // Block size in uint of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of datatype
  const uint32_t threadSize = THREAD_SIZE;

  // Allocate shared memory for the lookahead buffer of the whole block, the
  // sliding window is included

  // initialize the tid
  int tid = 0;

  // Copy the memeory from global to shared
  for (int i = 0; i < blockSize / threadSize; i++)
  {
    buffer[item.get_local_id(2) + threadSize * i] =
        input[item.get_group(2) * blockSize + item.get_local_id(2) +
              threadSize * i];
  }

  // Synchronize all threads to ensure that the buffer is fully loaded
  item.barrier(sycl::access::fence_space::local_space);

  // find match for every data point
  for (int iteration = 0; iteration < (int)(blockSize / threadSize);
       iteration++)
  {
    // Initialize the lookahead buffer and the sliding window pointers
    tid = item.get_local_id(2) + iteration * threadSize;
    int bufferStart = tid;
    int bufferPointer = bufferStart;
    int windowStart =
        bufferStart - int(WINDOW_SIZE) < 0 ? 0 : bufferStart - WINDOW_SIZE;
    int windowPointer = windowStart;

    uint8_t maxLen = 0;
    uint8_t maxOffset = 0;
    uint8_t len = 0;
    uint8_t offset = 0;

    while (windowPointer < bufferStart && bufferPointer < blockSize)
    {
      if (buffer[bufferPointer] == buffer[windowPointer])
      {
        if (offset == 0)
        {
          offset = bufferPointer - windowPointer;
        }
        len++;
        bufferPointer++;
      }
      else
      {
        if (len > maxLen)
        {
          maxLen = len;
          maxOffset = offset;
        }
        len = 0;
        offset = 0;
        bufferPointer = bufferStart;
      }
      windowPointer++;
    }
    if (len > maxLen)
    {
      maxLen = len;
      maxOffset = offset;
    }

    lengthBuffer[item.get_local_id(2) + iteration * threadSize] = maxLen;
    offsetBuffer[item.get_local_id(2) + iteration * threadSize] = maxOffset;

    // initialize array as 0
    prefixBuffer[item.get_local_id(2) + iteration * threadSize] = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);

  // find encode information
  uint32_t flagCount = 0;

  if (item.get_local_id(2) == 0)
  {
    uint8_t flagPosition = 0x01;
    uint8_t byteFlag = 0;

    int encodeIndex = 0;

    while (encodeIndex < blockSize)
    {
      // if length < minEncodeLength, no match is found
      if (lengthBuffer[encodeIndex] < minEncodeLength)
      {
        prefixBuffer[encodeIndex] = sizeof(T);
        encodeIndex++;
      }
      // if length > minEncodeLength, match is found
      else
      {
        prefixBuffer[encodeIndex] = 2;
        encodeIndex += lengthBuffer[encodeIndex];
        byteFlag |= flagPosition;
      }
      // store the flag if there are 8 bits already
      if (flagPosition == 0x80)
      {
        byteFlagArr[flagCount] = byteFlag;
        flagCount++;
        flagPosition = 0x01;
        byteFlag = 0;
        continue;
      }
      flagPosition <<= 1;
    }
    if (flagPosition != 0x01)
    {
      byteFlagArr[flagCount] = byteFlag;
      flagCount++;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  // prefix summation, up-sweep
  int prefixSumOffset = 1;
  for (uint32_t d = blockSize >> 1; d > 0; d = d >> 1)
  {
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++)
    {
      tid = item.get_local_id(2) + iteration * threadSize;
      if (tid < d)
      {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;
        prefixBuffer[bi] += prefixBuffer[ai];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    prefixSumOffset *= 2;
  }

  // clear the last element
  if (item.get_local_id(2) == 0)
  {
    // printf("block size: %d flag array size: %d\n", prefixBuffer[blockSize - 1], flagCount);
    compressedDataSizeGlobal[item.get_group(2)] =
        prefixBuffer[blockSize - 1];
    flagArrSizeGlobal[item.get_group(2)] = flagCount;
    prefixBuffer[blockSize] = prefixBuffer[blockSize - 1];
    prefixBuffer[blockSize - 1] = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);

  // prefix summation, down-sweep
  for (int d = 1; d < blockSize; d *= 2)
  {
    prefixSumOffset >>= 1;
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++)
    {
      tid = item.get_local_id(2) + iteration * threadSize;

      if (tid < d)
      {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;

        uint32_t t = prefixBuffer[ai];
        prefixBuffer[ai] = prefixBuffer[bi];
        prefixBuffer[bi] += t;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
  }

  // encoding phase one
  int tmpCompressedDataGlobalOffset;
  tmpCompressedDataGlobalOffset = blockSize * item.get_group(2) * sizeof(T);
  for (int iteration = 0; iteration < (int)(blockSize / threadSize); iteration++)
  {
    tid = item.get_local_id(2) + iteration * threadSize;
    if (prefixBuffer[tid + 1] != prefixBuffer[tid])
    {
      if (lengthBuffer[tid] < minEncodeLength)
      {
        uint32_t tmpOffset = prefixBuffer[tid];
        uint8_t *bytePtr = (uint8_t *)&buffer[tid];
        for (int tmpIndex = 0; tmpIndex < sizeof(T); tmpIndex++)
        {
          tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset + tmpIndex] = *(bytePtr + tmpIndex);
        }
      }
      else
      {
        uint32_t tmpOffset = prefixBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset] = lengthBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset + 1] = offsetBuffer[tid];
      }
    }
  }

  // Copy the memeory back
  if (item.get_local_id(2) == 0)
  {
    for (int flagArrIndex = 0; flagArrIndex < flagCount; flagArrIndex++)
    {
      tmpFlagArrGlobal[blockSize / 8 * item.get_group(2) + flagArrIndex] =
          byteFlagArr[flagArrIndex];
    }
  }
}

// Define the compress Encode kernel functions
template <typename T>
void compressKernelIII(uint32_t numOfBlocks,
                       const uint32_t *__restrict flagArrOffsetGlobal,
                       const uint32_t *__restrict compressedDataOffsetGlobal,
                       const uint8_t *__restrict tmpFlagArrGlobal,
                       const uint8_t *__restrict tmpCompressedDataGlobal,
                       uint8_t *__restrict flagArrGlobal,
                       uint8_t *__restrict compressedDataGlobal,
                       const sycl::nd_item<3> &item)
{
  // Block size in uint of bytes
  const int blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of bytes
  const int threadSize = THREAD_SIZE;

  // find block index
  int blockIndex = item.get_group(2);

  int flagArrOffset = flagArrOffsetGlobal[blockIndex];
  int flagArrSize = flagArrOffsetGlobal[blockIndex + 1] - flagArrOffsetGlobal[blockIndex];

  int compressedDataOffset = compressedDataOffsetGlobal[blockIndex];
  int compressedDataSize = compressedDataOffsetGlobal[blockIndex + 1] - compressedDataOffsetGlobal[blockIndex];

  int tid = item.get_local_id(2);

  while (tid < flagArrSize)
  {
    flagArrGlobal[flagArrOffset + tid] = tmpFlagArrGlobal[blockSize / 8 * blockIndex + tid];
    tid += threadSize;
  }

  tid = item.get_local_id(2);

  while (tid < compressedDataSize)
  {
    compressedDataGlobal[compressedDataOffset + tid] = tmpCompressedDataGlobal[blockSize * sizeof(T) * blockIndex + tid];
    tid += threadSize;
  }
}

// Define the decompress kernel functions
template <typename T>
void decompressKernel(T *output, uint32_t numOfBlocks,
                      const uint32_t *__restrict flagArrOffsetGlobal,
                      const uint32_t *__restrict compressedDataOffsetGlobal,
                      const uint8_t *__restrict flagArrGlobal,
                      const uint8_t *__restrict compressedDataGlobal,
                      const sycl::nd_item<3> &item)
{
  // Block size in unit of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  int tid = item.get_group(2) * item.get_local_range(2) +
            item.get_local_id(2);

  if (tid < numOfBlocks)
  {
    int flagArrOffset = flagArrOffsetGlobal[tid];
    int flagArrSize = flagArrOffsetGlobal[tid + 1] - flagArrOffsetGlobal[tid];

    int compressedDataOffset = compressedDataOffsetGlobal[tid];

    uint32_t dataPointsIndex = 0;
    uint32_t compressedDataIndex = 0;

    uint8_t byteFlag;

    for (int flagArrayIndex = 0; flagArrayIndex < flagArrSize; flagArrayIndex++)
    {
      byteFlag = flagArrGlobal[flagArrOffset + flagArrayIndex];

      for (int bitCount = 0; bitCount < 8; bitCount++)
      {
        int matchFlag = (byteFlag >> bitCount) & 0x1;
        if (matchFlag == 1)
        {
          int length = compressedDataGlobal[compressedDataOffset + compressedDataIndex];
          int offset = compressedDataGlobal[compressedDataOffset + compressedDataIndex + 1];
          compressedDataIndex += 2;
          int dataPointsStart = dataPointsIndex;
          for (int tmpDecompIndex = 0; tmpDecompIndex < length; tmpDecompIndex++)
          {
            output[tid * blockSize + dataPointsIndex] = output[tid * blockSize + dataPointsStart - offset + tmpDecompIndex];
            dataPointsIndex++;
          }
        }
        else
        {
          uint8_t *tmpPtr = (uint8_t *)&output[tid * blockSize + dataPointsIndex];
          for (int tmpDecompIndex = 0; tmpDecompIndex < sizeof(T); tmpDecompIndex++)
          {
            *(tmpPtr + tmpDecompIndex) = compressedDataGlobal[compressedDataOffset + compressedDataIndex + tmpDecompIndex];
          }

          compressedDataIndex += sizeof(T);
          dataPointsIndex++;
        }
        if (dataPointsIndex >= blockSize)
        {
          return;
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  std::string inputFileName;
  int opt;

  /* parse command line */
  while ((opt = getopt(argc, argv, "i:h")) != -1)
  {
    switch (opt)
    {
    case 'i': /* input file name */
      inputFileName = optarg;
      break;

    case 'h': /* help */
      printf(" Usage for compression and decompression: ./main -i {inputfile}\n");
      return 0;
    }
  }

  INPUT_TYPE *hostArray = io::read_binary_to_new_array<INPUT_TYPE>(inputFileName);

#ifdef DEBUG
  int debugOffset = 0;

  printf("print the first 1024 elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    std::cout << hostArray[tmpIndex + debugOffset] << "\t";
  }
  printf("\n");
#endif

  INPUT_TYPE *deviceArray;
  INPUT_TYPE *deviceOutput;
  uint32_t fileSize = io::FileSize(inputFileName);

  uint32_t *flagArrSizeGlobal;
  uint32_t *flagArrOffsetGlobal;
  uint32_t *compressedDataSizeGlobal;
  uint32_t *compressedDataOffsetGlobal;
  uint8_t *tmpFlagArrGlobal;
  uint8_t *tmpCompressedDataGlobal;
  uint8_t *flagArrGlobal;
  uint8_t *compressedDataGlobal;

  // calculate the padding size, unit in bytes
  uint32_t paddingSize = fileSize % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE - fileSize % BLOCK_SIZE;

  // calculate the datatype size, unit in datatype
  uint32_t datatypeSize =
      static_cast<uint32_t>((fileSize + paddingSize) / sizeof(INPUT_TYPE));

  uint32_t numOfBlocks = datatypeSize * sizeof(INPUT_TYPE) / BLOCK_SIZE;

  INPUT_TYPE *hostOutput = (INPUT_TYPE *)malloc(sizeof(INPUT_TYPE) * datatypeSize);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // malloc the device buffer and set it as 0
  deviceArray = (uint32_t *)sycl::malloc_device(fileSize + paddingSize, q);
  deviceOutput = (uint32_t *)sycl::malloc_device(fileSize + paddingSize, q);

  flagArrSizeGlobal = sycl::malloc_device<uint32_t>((numOfBlocks + 1), q);
  flagArrOffsetGlobal = sycl::malloc_device<uint32_t>((numOfBlocks + 1), q);
  compressedDataSizeGlobal = sycl::malloc_device<uint32_t>((numOfBlocks + 1), q);
  compressedDataOffsetGlobal = sycl::malloc_device<uint32_t>((numOfBlocks + 1), q);
  tmpFlagArrGlobal = sycl::malloc_device<uint8_t>(datatypeSize / 8, q);
  tmpCompressedDataGlobal = (uint8_t*) sycl::malloc_device(sizeof(INPUT_TYPE) * datatypeSize, q);
  flagArrGlobal = sycl::malloc_device<uint8_t>(datatypeSize / 8, q);
  compressedDataGlobal = (uint8_t*) sycl::malloc_device(sizeof(INPUT_TYPE) * datatypeSize, q);

  // initialize the mem as 0
  q.memset(deviceArray, 0, fileSize + paddingSize);
  q.memset(deviceOutput, 0, fileSize + paddingSize);
  q.memset(flagArrSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  q.memset(flagArrOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  q.memset(compressedDataSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  q.memset(compressedDataOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  q.memset(tmpFlagArrGlobal, 0, sizeof(uint8_t) * datatypeSize / 8);
  q.memset(tmpCompressedDataGlobal, 0, sizeof(INPUT_TYPE) * datatypeSize);

  // copy host memory to device
  q.memcpy(deviceArray, hostArray, fileSize);

  // printf("num of blocks: %d\nfile size: %d\npadding size: %d\n data type length: %d\n", numOfBlocks, fileSize, paddingSize, datatypeSize);

  sycl::range<3> gridDim(1, 1, numOfBlocks);
  sycl::range<3> blockDim(1, 1, THREAD_SIZE);

  sycl::range<3> deGridDim(1, 1, ceil(float(numOfBlocks) / 32));
  sycl::range<3> deBlockDim(1, 1, 32);

  uint32_t *flagArrOffsetGlobalHost;
  uint32_t *compressedDataOffsetGlobalHost;
  uint8_t *tmpFlagArrGlobalHost;
  uint8_t *tmpCompressedDataGlobalHost;
  uint8_t *flagArrGlobalHost;
  uint8_t *compressedDataGlobalHost;

  flagArrOffsetGlobalHost = (uint32_t *)malloc(sizeof(uint32_t) * (numOfBlocks + 1));
  compressedDataOffsetGlobalHost = (uint32_t *)malloc(sizeof(uint32_t) * (numOfBlocks + 1));
  tmpFlagArrGlobalHost = (uint8_t *)malloc(sizeof(uint8_t) * datatypeSize / 8);
  tmpCompressedDataGlobalHost = (uint8_t *)malloc(sizeof(INPUT_TYPE) * datatypeSize);
  flagArrGlobalHost = (uint8_t *)malloc(sizeof(uint8_t) * datatypeSize / 8);
  compressedDataGlobalHost = (uint8_t *)malloc(sizeof(INPUT_TYPE) * datatypeSize);

  int minEncodeLength = sizeof(INPUT_TYPE) == 1 ? 2 : 1;

  const uint32_t blockSize = BLOCK_SIZE / sizeof(INPUT_TYPE);

  q.wait();
  auto compStart = std::chrono::steady_clock::now();

  // launch kernels
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<INPUT_TYPE, 1> buffer_acc(
        sycl::range<1>(blockSize), cgh);
    sycl::local_accessor<uint8_t, 1> lengthBuffer_acc(
        sycl::range<1>(blockSize), cgh);
    sycl::local_accessor<uint8_t, 1> offsetBuffer_acc(
        sycl::range<1>(blockSize), cgh);
    sycl::local_accessor<uint32_t, 1> prefixBuffer_acc(
        sycl::range<1>(blockSize + 1), cgh);
    sycl::local_accessor<uint8_t, 1> byteFlagArr_acc(
        sycl::range<1>((blockSize / 8)), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(gridDim * blockDim, blockDim),
        [=](sycl::nd_item<3> item) {
          compressKernelI<INPUT_TYPE>(
              deviceArray, numOfBlocks, flagArrSizeGlobal,
              compressedDataSizeGlobal, tmpFlagArrGlobal,
              tmpCompressedDataGlobal, minEncodeLength, item,
              buffer_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
              lengthBuffer_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
              offsetBuffer_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
              prefixBuffer_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
              byteFlagArr_acc.get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });

  // Determine temporary device storage requirements

  // Run exclusive prefix sum
  oneapi::dpl::exclusive_scan(
      oneapi::dpl::execution::device_policy(q), flagArrSizeGlobal,
      flagArrSizeGlobal + numOfBlocks + 1, flagArrOffsetGlobal,
      typename std::iterator_traits<decltype(flagArrSizeGlobal)>::value_type{});

  // Determine temporary device storage requirements

  // Run exclusive prefix sum
  oneapi::dpl::exclusive_scan(
      oneapi::dpl::execution::device_policy(q), compressedDataSizeGlobal,
      compressedDataSizeGlobal + numOfBlocks + 1, compressedDataOffsetGlobal,
      typename std::iterator_traits<
          decltype(compressedDataSizeGlobal)>::value_type{});

  q.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
      [=](sycl::nd_item<3> item) {
        compressKernelIII<INPUT_TYPE>(
            numOfBlocks, flagArrOffsetGlobal, compressedDataOffsetGlobal,
            tmpFlagArrGlobal, tmpCompressedDataGlobal, flagArrGlobal,
            compressedDataGlobal, item);
      });

  q.wait();
  auto compStop = std::chrono::steady_clock::now();

  auto decompStart = std::chrono::steady_clock::now();

  q.parallel_for(sycl::nd_range<3>(deGridDim * deBlockDim, deBlockDim),
                 [=](sycl::nd_item<3> item) {
                           decompressKernel<INPUT_TYPE>(
                               deviceOutput, numOfBlocks, flagArrOffsetGlobal,
                               compressedDataOffsetGlobal, flagArrGlobal,
                               compressedDataGlobal, item);
                         });
  q.wait();
  auto decompStop = std::chrono::steady_clock::now();

  // copy the memory back to global
  q.memcpy(flagArrOffsetGlobalHost, flagArrOffsetGlobal,
               sizeof(uint32_t) * (numOfBlocks + 1));
  q.memcpy(compressedDataOffsetGlobalHost, compressedDataOffsetGlobal,
               sizeof(uint32_t) * (numOfBlocks + 1));
  q.memcpy(tmpFlagArrGlobalHost, tmpFlagArrGlobal,
               sizeof(uint8_t) * datatypeSize / 8);
  q.memcpy(tmpCompressedDataGlobalHost, tmpCompressedDataGlobal,
               sizeof(INPUT_TYPE) * datatypeSize);
  q.memcpy(flagArrGlobalHost, flagArrGlobal,
               sizeof(uint8_t) * datatypeSize / 8);
  q.memcpy(compressedDataGlobalHost, compressedDataGlobal,
               sizeof(INPUT_TYPE) * datatypeSize);

  q.memcpy(hostOutput, deviceOutput, fileSize);
  q.wait();

#ifdef DEBUG
  printf("print the first 1024 flag array offset elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", flagArrOffsetGlobalHost[tmpIndex]);
  }
  printf("\n");

  printf("print the first 1024 compressed data offset elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", compressedDataOffsetGlobalHost[tmpIndex]);
  }
  printf("\n");

  printf("print the first 1024 flag array elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", flagArrGlobalHost[tmpIndex]);
  }
  printf("\n");

  printf("print the first 1024 compressed data elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", compressedDataGlobalHost[tmpIndex]);
  }
  printf("\n");

  // printf("print the first 1024 tmp flag array elements:\n");
  // for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  // {
  //   printf("%d\t", tmpFlagArrGlobalHost[tmpIndex]);
  // }
  // printf("\n");
#endif

  // verify the final output
  for (int verifyIndex = 0; verifyIndex < fileSize / sizeof(INPUT_TYPE); verifyIndex++)
  {
    if (hostArray[verifyIndex] != hostOutput[verifyIndex])
    {
      printf("verification failed!!! Index %d is wrong\n", verifyIndex);
      std::cout << "hostArray: " << hostArray[verifyIndex] << ", hostOutput: " << hostOutput[verifyIndex] << std::endl;
      break;
    }
  }

  float originalSize = fileSize;
  float compressedSize = sizeof(uint32_t) * (numOfBlocks + 1) * 2 + flagArrOffsetGlobalHost[numOfBlocks] + compressedDataOffsetGlobalHost[numOfBlocks];
  float compressionRatio = originalSize / compressedSize;
  std::cout << "compression ratio: " << compressionRatio << std::endl;

  float compTime = std::chrono::duration<float, std::milli>(compStop - compStart).count();
  float decompTime = std::chrono::duration<float, std::milli>(decompStop - decompStart).count();
  float compTp = float(fileSize) / 1024 / 1024 / compTime;
  float decompTp = float(fileSize) / 1024 / 1024 / decompTime;
  std::cout << "compression e2e throughput: " << compTp << " GB/s" << std::endl;
  std::cout << "decompression e2e throughput: " << decompTp << " GB/s" << std::endl;

  // free dynamic arrays
  free(flagArrOffsetGlobalHost);
  free(compressedDataOffsetGlobalHost);
  free(tmpFlagArrGlobalHost);
  free(tmpCompressedDataGlobalHost);
  free(flagArrGlobalHost);
  free(compressedDataGlobalHost);

  sycl::free(deviceArray, q);
  sycl::free(deviceOutput, q);

  sycl::free(flagArrSizeGlobal, q);
  sycl::free(flagArrOffsetGlobal, q);
  sycl::free(compressedDataSizeGlobal, q);
  sycl::free(compressedDataOffsetGlobal, q);
  sycl::free(tmpFlagArrGlobal, q);
  sycl::free(tmpCompressedDataGlobal, q);
  sycl::free(flagArrGlobal, q);
  sycl::free(compressedDataGlobal, q);

  free(hostOutput);

  delete hostArray;

  return 0;
}
