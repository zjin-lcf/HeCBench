#include <cmath>
#include <algorithm>
#include "utils.h"

// DEVICE FUNCTIONS

int clamp(size_t const val, int const max)
{
  return static_cast<int>(
      sycl::min((unsigned long)static_cast<size_t>(max), (unsigned long)val));
}

template <typename T>
void readMinAndMax(
    T const* const inMin,
    T const* const inMax,
    T* const minBuffer,
    T* const maxBuffer,
    int const blockOffset,
    int const blockEnd,
    sycl::nd_item<1> &item)
{
  static_assert(
      BLOCK_SIZE <= BLOCK_WIDTH,
      "BLOCK_SIZE must be less than or equal to BLOCK_WIDTH");

  int lid = item.get_local_id(0);

  if (lid < blockEnd) {
    T localMin = inMin[blockOffset + lid];
    T localMax = inMax[blockOffset + lid];
    for (int i = lid + BLOCK_SIZE;
         i < BLOCK_WIDTH && i < blockEnd; i += BLOCK_SIZE) {
      int const readIdx = blockOffset + i;
      localMin = sycl::min(inMin[readIdx], localMin);
      localMax = sycl::max(inMax[readIdx], localMax);
    }
    minBuffer[lid] = localMin;
    maxBuffer[lid] = localMax;
  }
}

template <typename T>
void
reduceMinAndMax(T* const minBuffer, T* const maxBuffer, int const blockEnd,
                sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);

  // cooperatively compute min and max
  for (int d = BLOCK_SIZE / 2; d > 0; d >>= 1) {
    if (lid < BLOCK_SIZE / 2) {
      int const idx = lid;
      if (idx < d && idx + d < blockEnd) {
        minBuffer[idx] = sycl::min(minBuffer[idx], minBuffer[d + idx]);
      }
    } else {
      int const idx = lid - (BLOCK_SIZE / 2);
      if (idx < d && idx + d < blockEnd) {
        maxBuffer[idx] = sycl::max(maxBuffer[idx], maxBuffer[d + idx]);
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}


// KERNELS

template <typename LIMIT, typename INPUT>
void bitPackConfigScanKernel(
    LIMIT* const minValue,
    LIMIT* const maxValue,
    INPUT const* const in,
    const size_t* const numDevice,
    sycl::nd_item<1> &item,
    LIMIT *minBuffer,
    LIMIT *maxBuffer)
{
  static_assert(BLOCK_SIZE % 64 == 0, "BLOCK_SIZE must a multiple of 64");

  //assert(BLOCK_SIZE == item.get_local_range(0));

  const size_t num = *numDevice;
  const int numBlocks = roundUpDiv(num, BLOCK_SIZE);

  //assert(num > 0);

  int lid = item.get_local_id(0);
  //assert(lid < BLOCK_SIZE);

  int bid = item.get_group(0);

  if (bid < numBlocks) {
    // each block processes it's chunks, updates min/max

    LIMIT localMin = 0;
    LIMIT localMax = 0;

    int lastThread = 0;
    for (int block = bid; block < numBlocks;
         block += item.get_group_range(0)) {

      int const blockOffset = BLOCK_SIZE * block;
      int const blockEnd =
          sycl::min((int)(static_cast<int>(num) - blockOffset), BLOCK_SIZE);

      lastThread = sycl::max(lastThread, blockEnd);

      if (lid < blockEnd) {
        LIMIT const val = in[blockOffset + lid];
        if (block == bid) {
          // first iteration just set values
          localMax = val;
          localMin = val;
        } else {
          localMin = sycl::min(val, localMin);
          localMax = sycl::max(val, localMax);
        }
      }
    }

    minBuffer[lid] = localMin;
    maxBuffer[lid] = localMax;

    item.barrier(sycl::access::fence_space::local_space);

    // cooperatively compute min and max
    reduceMinAndMax(minBuffer, maxBuffer, lastThread, item);

    if (lid == 0) {
      minValue[bid] = minBuffer[0];
      maxValue[bid] = maxBuffer[0];
    }
  }
}

template <typename LIMIT, typename INPUT>
void bitPackConfigFinalizeKernel(
    LIMIT const* const inMin,
    LIMIT const* const inMax,
    unsigned char* const numBitsPtr,
    INPUT* const outMinValPtr,
    const size_t* const numDevice,
    sycl::nd_item<1> &item,
    LIMIT *minBuffer,
    LIMIT *maxBuffer)
{
  static_assert(
      BLOCK_SIZE <= BLOCK_WIDTH,
      "BLOCK_SIZE must be less than or equal to BLOCK_WIDTH");
  static_assert(
      BLOCK_WIDTH % BLOCK_SIZE == 0,
      "BLOCK_WIDTH must be a multiple of BLOCK_SIZE");
  static_assert(BLOCK_SIZE % 64 == 0, "BLOCK_SIZE must a multiple of 64");

  //int bid = item.get_group(0);
  //assert(bid == 0);

  const size_t num = sycl::min(roundUpDiv(*numDevice, BLOCK_SIZE),
                               static_cast<size_t>(BLOCK_WIDTH));

  //assert(num > 0);

  int lid = item.get_local_id(0);

  // each block processes it's chunk, updates min/max, and the calculates
  // the bitwidth based on the last update

  // load data
  readMinAndMax(inMin, inMax, minBuffer, maxBuffer, 0, num, item);

  item.barrier(sycl::access::fence_space::local_space);

  // cooperatively compute min and max
  reduceMinAndMax(minBuffer, maxBuffer, sycl::min(BLOCK_SIZE, (int)num), item);

  if (lid == 0) {
    *outMinValPtr = static_cast<INPUT>(minBuffer[0]);
    // we need to update the number of bits
    if (sizeof(LIMIT) > sizeof(int)) {
      const uint64_t range = static_cast<uint64_t>(maxBuffer[0]) - static_cast<uint64_t>(minBuffer[0]);
      // need 64 bit clz
      *numBitsPtr = sizeof(uint64_t) * 8 - sycl::clz(range);
    } else {
      const uint32_t range = static_cast<uint32_t>(maxBuffer[0]) - static_cast<uint32_t>(minBuffer[0]);
      // can use 32 bit clz
      *numBitsPtr = sizeof(uint32_t) * 8 - sycl::clz(range);
    }
  }
}

template <typename INPUT, typename OUTPUT>
void bitPackKernel(
    unsigned char const* const numBitsPtr,
    INPUT const* const valueOffsetPtr,
    OUTPUT* const outPtr,
    INPUT const* const in,
    const size_t* const numDevice,
    sycl::nd_item<1> &item,
    typename std::make_unsigned<INPUT>::type *inBuffer)
{
  const size_t num = *numDevice;

  const int numBlocks = roundUpDiv(num, BLOCK_SIZE);

  OUTPUT* const out = outPtr;
  int const numBits = *numBitsPtr;
  INPUT const valueOffset = *valueOffsetPtr;

  int lid = item.get_local_id(0);

  for (int blockId = item.get_group(0); blockId < numBlocks;
       blockId += item.get_group_range(0)) {
    // The kernel works by assigning an output index to each thread.
    // The kernel then iterates over chunks of input, filling the bits
    // for each thread.
    // And then writing the stored bits to the output.
    int const outputIdx = lid + blockId * BLOCK_SIZE;
    //assert(outputIdx >= 0);
    //assert(*numBitsPtr <= sizeof(INPUT) * 8U);

    size_t const bitStart = outputIdx * sizeof(*out) * 8U;
    size_t const bitEnd = bitStart + (sizeof(*out) * 8U);

    int const startIdx = clamp(bitStart / static_cast<size_t>(numBits), num);
    int const endIdx = clamp(roundUpDiv(bitEnd, numBits), num);
    //assert(startIdx >= 0);

    size_t const blockStartBit = blockId * BLOCK_SIZE * sizeof(*out) * 8U;
    size_t const blockEndBit = (blockId + 1) * BLOCK_SIZE * sizeof(*out) * 8U;
    //assert(blockStartBit < blockEndBit);

    int const blockStartIdx = clamp(
        roundDownTo(blockStartBit / static_cast<size_t>(numBits), BLOCK_SIZE),
        num);
    int const blockEndIdx
        = clamp(roundUpTo(roundUpDiv(blockEndBit, numBits), BLOCK_SIZE), num);
    //assert(blockStartIdx >= 0);
    //assert(blockStartIdx <= blockEndIdx);

    OUTPUT val = 0;
    for (int bufferStart = blockStartIdx; bufferStart < blockEndIdx;
         bufferStart += BLOCK_SIZE) {
      item.barrier(sycl::access::fence_space::local_space);

      // fill input buffer
      int const inputIdx = bufferStart + lid;
      if (inputIdx < num) {
        inBuffer[lid] = in[inputIdx] - valueOffset;
      }

      item.barrier(sycl::access::fence_space::local_space);

      int const currentStartIdx = sycl::max(startIdx, bufferStart);
      int const currentEndIdx =
          sycl::min(endIdx, (int)(bufferStart + BLOCK_SIZE));

      for (int idx = currentStartIdx; idx < currentEndIdx; ++idx) {
        int const localIdx = idx - bufferStart;

        // keep only bits we're interested in
        OUTPUT bits = static_cast<OUTPUT>(inBuffer[localIdx]);
        int const offset = static_cast<int>(
            static_cast<ssize_t>(idx * numBits)
            - static_cast<ssize_t>(bitStart));
        //assert(sycl::abs(offset) < sizeof(bits) * 8U);

        if (offset > 0) {
          bits <<= offset;
        } else {
          bits >>= -offset;
        }

        // update b
        val |= bits;
      }
    }

    if (startIdx < num) {
      out[outputIdx] = val;
    }
  }
}


/**
 * @brief Launch of all of the kernels necessary for the configuration step of
 * bit packing.
 *
 * @tparam LIMIT The type used for min/max values.
 * @tparam INPUT The type being reduced.
 * @param minValueScratch Space used by the kernels to reduce the minimum
 * values. Must be at least the size returned by `getReduceScratchSpaceSize()`.
 * @param maxValueScratch Space used by the kernels to reduce the maximum
 * values. Must be at least the size returned by `getReduceScratchSpaceSize()`.
 * @param minValOutPtr The place to put the actual minimum value of the entire
 * series (output).
 * @param numBitsPtr The number of bits to compact to (output).
 * @param in The input to be compressed.
 * @param numDevice The number of elements on the device.
 * @param maxNum The maximum number of elements in the input.
 */
template <typename LIMIT, typename INPUT>
void bitPackConfigLaunch(
    sycl::queue &q,
    LIMIT* const minValueScratch,
    LIMIT* const maxValueScratch,
    INPUT* const minValOutPtr,
    unsigned char* const numBitsPtr,
    INPUT const* const in,
    const size_t* const numDevice,
    size_t const maxNum)
{
  const sycl::range<1> grid(std::min(BLOCK_WIDTH, static_cast<int>(roundUpDiv(maxNum, BLOCK_SIZE))));
  const sycl::range<1> block(BLOCK_SIZE);

  // make sure the result will fit in a single block for the finalize kernel
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<LIMIT, 1> minBuffer(sycl::range<1>(BLOCK_SIZE), cgh);
    sycl::local_accessor<LIMIT, 1> maxBuffer(sycl::range<1>(BLOCK_SIZE), cgh);
    cgh.parallel_for(sycl::nd_range<1>(grid * block, block), [=](sycl::nd_item<1> item) {
      bitPackConfigScanKernel(minValueScratch, maxValueScratch, in, numDevice, item,
                              (LIMIT *)minBuffer.get_pointer(),
                              (LIMIT *)maxBuffer.get_pointer());
      });
  });

  // determine numBits and convert min value
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<LIMIT, 1> minBuffer(sycl::range<1>(BLOCK_SIZE), cgh);
    sycl::local_accessor<LIMIT, 1> maxBuffer(sycl::range<1>(BLOCK_SIZE), cgh);
    cgh.parallel_for(sycl::nd_range<1>(block, block), [=](sycl::nd_item<1> item) {
      bitPackConfigFinalizeKernel(
         minValueScratch, maxValueScratch, numBitsPtr, minValOutPtr,
         numDevice, item, (LIMIT *)minBuffer.get_pointer(),
         (LIMIT *)maxBuffer.get_pointer());
    });
  });
}

template <typename INPUT, typename OUTPUT>
void bitPackLaunch(
    sycl::queue &q,
    const INPUT * const minValueDevicePtr,
    unsigned char const* const numBitsDevicePtr,
    OUTPUT* const outPtr,
    INPUT const* const in,
    const size_t* const numDevice,
    const size_t maxNum)
{
  static_assert(BLOCK_SIZE % (sizeof(OUTPUT) * 8U) == 0,
      "Block size must be a multiple of output word size.");

  sycl::range<1> const gws (std::min(4096, static_cast<int>(roundUpDiv(maxNum, BLOCK_SIZE))) * BLOCK_SIZE);
  sycl::range<1> const lws (BLOCK_SIZE);

  using UINPUT = typename std::make_unsigned<INPUT>::type;
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<UINPUT, 1> inBuffer(sycl::range<1>(256), cgh);
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      bitPackKernel(numBitsDevicePtr, minValueDevicePtr, outPtr, in,
                    numDevice, item, inBuffer.get_pointer());
    });
  });
}

template <typename IN, typename OUT, typename LIMIT>
void bitPackFixedBitAndMinInternal(
    sycl::queue &q,
    void const* const minValueDevicePtr,
    unsigned char const* const numBitsDevicePtr,
    void* const /* workspace */,
    void* const outPtr,
    void const* const in,
    const size_t* const numDevice,
    size_t const maxNum)
{
  // type qualifier is meaningless on cast type
  OUT* const outputTypedPtr = reinterpret_cast<OUT*>(outPtr);
  IN const* const inputTyped = static_cast<IN const*>(in);

  bitPackLaunch(
      q,  
      reinterpret_cast<const IN*>(minValueDevicePtr),
      numBitsDevicePtr,
      outputTypedPtr,
      inputTyped,
      numDevice,
      maxNum);
}

template <typename IN, typename OUT, typename LIMIT>
void bitPackInternal(
    sycl::queue &q,
    void* const workspace,
    void* const outPtr,
    void const* const in,
    const size_t* const numDevice,
    size_t const maxNum,
    void* const minValueDevicePtr,
    unsigned char* const numBitsDevicePtr)
{
  // cast voids to known types
  LIMIT* const maxValueTyped = static_cast<LIMIT*>(workspace);
  LIMIT* const minValueTyped = maxValueTyped + getReduceScratchSpaceSize(maxNum);
  IN const* const inputTyped = static_cast<IN const*>(in);

  // determine min, and bit width
  bitPackConfigLaunch(
      q,
      minValueTyped,
      maxValueTyped,
      reinterpret_cast<IN*>(minValueDevicePtr),
      numBitsDevicePtr,
      inputTyped,
      numDevice,
      maxNum);

  bitPackFixedBitAndMinInternal<IN, OUT, LIMIT>(
      q,
      minValueDevicePtr,
      numBitsDevicePtr,
      workspace,
      outPtr,
      in,
      numDevice,
      maxNum);
}


void compress(
    sycl::queue &q,
    void* const workspace,
    const size_t workspaceSize,
    const nvcompType_t inType,
    void* const outPtr,
    const void* const in,
    const size_t* const numPtr,
    const size_t maxNum,
    void* const minValueDevicePtr,
    unsigned char* const numBitsDevicePtr)
{
  const size_t reqWorkSize = requiredWorkspaceSize(maxNum, inType);
  if (workspaceSize < reqWorkSize) {
    throw std::runtime_error(
        "Insufficient workspace size: " + std::to_string(workspaceSize)
        + ", need " + std::to_string(reqWorkSize));
  }

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < 1000; n++)
    NVCOMP_TYPE_SWITCH(
      inType,
      bitPackInternal,
      q,
      workspace,
      outPtr,
      in,
      numPtr,
      maxNum,
      minValueDevicePtr,
      numBitsDevicePtr);

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (1000 iterations) = %f (s)\n", time * 1e-9f);
}
