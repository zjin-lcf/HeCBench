#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "utils.h"
#include <cmath>

#include <algorithm>

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
    sycl::nd_item<3> item_ct1)
{
  static_assert(
      BLOCK_SIZE <= BLOCK_WIDTH,
      "BLOCK_SIZE must be less than or equal to BLOCK_WIDTH");

  if (item_ct1.get_local_id(2) < blockEnd) {
    T localMin = inMin[blockOffset + item_ct1.get_local_id(2)];
    T localMax = inMax[blockOffset + item_ct1.get_local_id(2)];
    for (int i = item_ct1.get_local_id(2) + BLOCK_SIZE;
         i < BLOCK_WIDTH && i < blockEnd; i += BLOCK_SIZE) {
      int const readIdx = blockOffset + i;
      localMin = sycl::min(inMin[readIdx], localMin);
      localMax = sycl::max(inMax[readIdx], localMax);
    }
    minBuffer[item_ct1.get_local_id(2)] = localMin;
    maxBuffer[item_ct1.get_local_id(2)] = localMax;
  }
}

template <typename T>
void
reduceMinAndMax(T* const minBuffer, T* const maxBuffer, int const blockEnd,
                sycl::nd_item<3> item_ct1)
{
  // cooperatively compute min and max
  for (int d = BLOCK_SIZE / 2; d > 0; d >>= 1) {
    if (item_ct1.get_local_id(2) < BLOCK_SIZE / 2) {
      int const idx = item_ct1.get_local_id(2);
      if (idx < d && idx + d < blockEnd) {
        minBuffer[idx] = sycl::min(minBuffer[idx], minBuffer[d + idx]);
      }
    } else {
      int const idx = item_ct1.get_local_id(2) - (BLOCK_SIZE / 2);
      if (idx < d && idx + d < blockEnd) {
        maxBuffer[idx] = sycl::max(maxBuffer[idx], maxBuffer[d + idx]);
      }
    }
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}


// KERNELS

template <typename LIMIT, typename INPUT>
void bitPackConfigScanKernel(
    LIMIT* const minValue,
    LIMIT* const maxValue,
    INPUT const* const in,
    const size_t* const numDevice,
    sycl::nd_item<3> item_ct1,
    LIMIT *minBuffer,
    LIMIT *maxBuffer)
{
  static_assert(BLOCK_SIZE % 64 == 0, "BLOCK_SIZE must a multiple of 64");

  assert(BLOCK_SIZE == item_ct1.get_local_range().get(2));

  const size_t num = *numDevice;
  const int numBlocks = roundUpDiv(num, BLOCK_SIZE);

  assert(num > 0);
  assert(item_ct1.get_local_id(2) < BLOCK_SIZE);

  if (item_ct1.get_group(2) < numBlocks) {
    // each block processes it's chunks, updates min/max

    LIMIT localMin = 0;
    LIMIT localMax = 0;

    int lastThread = 0;
    for (int block = item_ct1.get_group(2); block < numBlocks;
         block += item_ct1.get_group_range(2)) {

      int const blockOffset = BLOCK_SIZE * block;
      int const blockEnd =
          sycl::min((int)(static_cast<int>(num) - blockOffset), BLOCK_SIZE);

      lastThread = sycl::max(lastThread, blockEnd);

      if (item_ct1.get_local_id(2) < blockEnd) {
        LIMIT const val = in[blockOffset + item_ct1.get_local_id(2)];
        if (block == item_ct1.get_group(2)) {
          // first iteration just set values
          localMax = val;
          localMin = val;
        } else {
          localMin = sycl::min(val, localMin);
          localMax = sycl::max(val, localMax);
        }
      }
    }

    minBuffer[item_ct1.get_local_id(2)] = localMin;
    maxBuffer[item_ct1.get_local_id(2)] = localMax;

    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();

    // cooperatively compute min and max
    reduceMinAndMax(minBuffer, maxBuffer, lastThread, item_ct1);

    if (item_ct1.get_local_id(2) == 0) {
      minValue[item_ct1.get_group(2)] = minBuffer[0];
      maxValue[item_ct1.get_group(2)] = maxBuffer[0];
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
    sycl::nd_item<3> item_ct1,
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

  assert(item_ct1.get_group(2) == 0);

  const size_t num = sycl::min(roundUpDiv(*numDevice, BLOCK_SIZE),
                               (unsigned long)static_cast<size_t>(BLOCK_WIDTH));

  assert(num > 0);

  // each block processes it's chunk, updates min/max, and the calculates
  // the bitwidth based on the last update

  // load data
  readMinAndMax(inMin, inMax, minBuffer, maxBuffer, 0, num, item_ct1);

  /*
  DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  item_ct1.barrier();

  // cooperatively compute min and max
  reduceMinAndMax(minBuffer, maxBuffer, sycl::min(BLOCK_SIZE, (int)num),
                  item_ct1);

  if (item_ct1.get_local_id(2) == 0) {
    *outMinValPtr = static_cast<INPUT>(minBuffer[0]);
    // we need to update the number of bits
    if (sizeof(LIMIT) > sizeof(int)) {
      const long long int range = static_cast<uint64_t>(maxBuffer[0]) - static_cast<uint64_t>(minBuffer[0]);
      // need 64 bit clz
      *numBitsPtr = sizeof(long long int) * 8 - sycl::clz((long long)range);
    } else {
      const int range = static_cast<uint32_t>(maxBuffer[0]) - static_cast<uint32_t>(minBuffer[0]);
      // can use 32 bit clz
      *numBitsPtr = sizeof(int) * 8 - sycl::clz((int)range);
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
    sycl::nd_item<3> item_ct1,
    typename std::make_unsigned<INPUT>::type *inBuffer)
{
  const size_t num = *numDevice;

  const int numBlocks = roundUpDiv(num, BLOCK_SIZE);

  OUTPUT* const out = outPtr;
  int const numBits = *numBitsPtr;
  INPUT const valueOffset = *valueOffsetPtr;

  for (int blockId = item_ct1.get_group(2); blockId < numBlocks;
       blockId += item_ct1.get_group_range(2)) {
    // The kernel works by assigning an output index to each thread.
    // The kernel then iterates over chunks of input, filling the bits
    // for each thread.
    // And then writing the stored bits to the output.
    int const outputIdx = item_ct1.get_local_id(2) + blockId * BLOCK_SIZE;
    assert(outputIdx >= 0);
    assert(*numBitsPtr <= sizeof(INPUT) * 8U);

    size_t const bitStart = outputIdx * sizeof(*out) * 8U;
    size_t const bitEnd = bitStart + (sizeof(*out) * 8U);

    int const startIdx = clamp(bitStart / static_cast<size_t>(numBits), num);
    int const endIdx = clamp(roundUpDiv(bitEnd, numBits), num);
    assert(startIdx >= 0);

    size_t const blockStartBit = blockId * BLOCK_SIZE * sizeof(*out) * 8U;
    size_t const blockEndBit = (blockId + 1) * BLOCK_SIZE * sizeof(*out) * 8U;
    assert(blockStartBit < blockEndBit);

    int const blockStartIdx = clamp(
        roundDownTo(blockStartBit / static_cast<size_t>(numBits), BLOCK_SIZE),
        num);
    int const blockEndIdx
        = clamp(roundUpTo(roundUpDiv(blockEndBit, numBits), BLOCK_SIZE), num);
    assert(blockStartIdx >= 0);
    assert(blockStartIdx <= blockEndIdx);

    OUTPUT val = 0;
    for (int bufferStart = blockStartIdx; bufferStart < blockEndIdx;
         bufferStart += BLOCK_SIZE) {
      /*
      DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance, if there is no access to global memory.
      */
      item_ct1.barrier();

      // fill input buffer
      int const inputIdx = bufferStart + item_ct1.get_local_id(2);
      if (inputIdx < num) {
        inBuffer[item_ct1.get_local_id(2)] = in[inputIdx] - valueOffset;
      }

      /*
      DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance, if there is no access to global memory.
      */
      item_ct1.barrier();

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
        assert(sycl::abs(offset) < sizeof(bits) * 8U);

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
    LIMIT* const minValueScratch,
    LIMIT* const maxValueScratch,
    INPUT* const minValOutPtr,
    unsigned char* const numBitsPtr,
    INPUT const* const in,
    const size_t* const numDevice,
    size_t const maxNum)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  const sycl::range<3> grid(
      1, 1,
      std::min(BLOCK_WIDTH, static_cast<int>(roundUpDiv(maxNum, BLOCK_SIZE))));
  const sycl::range<3> block(1, 1, BLOCK_SIZE);

  // make sure the result will fit in a single block for the finalize kernel
  /*
  DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<LIMIT, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        minBuffer_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*/), cgh);
    sycl::accessor<LIMIT, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        maxBuffer_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          bitPackConfigScanKernel(minValueScratch, maxValueScratch, in,
                                  numDevice, item_ct1,
                                  (LIMIT *)minBuffer_acc_ct1.get_pointer(),
                                  (LIMIT *)maxBuffer_acc_ct1.get_pointer());
        });
  });

  // determine numBits and convert min value
  /*
  DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<LIMIT, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        minBuffer_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*/), cgh);
    sycl::accessor<LIMIT, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        maxBuffer_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(block, block), [=](sycl::nd_item<3> item_ct1) {
          bitPackConfigFinalizeKernel(
              minValueScratch, maxValueScratch, numBitsPtr, minValOutPtr,
              numDevice, item_ct1, (LIMIT *)minBuffer_acc_ct1.get_pointer(),
              (LIMIT *)maxBuffer_acc_ct1.get_pointer());
        });
  });
}

template <typename INPUT, typename OUTPUT>
void bitPackLaunch(
    const INPUT * const minValueDevicePtr,
    unsigned char const* const numBitsDevicePtr,
    OUTPUT* const outPtr,
    INPUT const* const in,
    const size_t* const numDevice,
    const size_t maxNum)
{
  static_assert(
      BLOCK_SIZE % (sizeof(OUTPUT) * 8U) == 0,
      "Block size must be a multiple of output word size.");

  sycl::range<3> const grid(
      1, 1, std::min(4096, static_cast<int>(roundUpDiv(maxNum, BLOCK_SIZE))));
  sycl::range<3> const block(1, 1, BLOCK_SIZE);

  /*
  DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */

  using UINPUT = typename std::make_unsigned<INPUT>::type;
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::accessor<UINPUT, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        inBuffer_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          bitPackKernel(numBitsDevicePtr, minValueDevicePtr, outPtr, in,
                        numDevice, item_ct1, inBuffer_acc_ct1.get_pointer());
        });
  });
}

template <typename IN, typename OUT, typename LIMIT>
void bitPackFixedBitAndMinInternal(
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
      reinterpret_cast<const IN*>(minValueDevicePtr),
      numBitsDevicePtr,
      outputTypedPtr,
      inputTyped,
      numDevice,
      maxNum);
}

template <typename IN, typename OUT, typename LIMIT>
void bitPackInternal(
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
      minValueTyped,
      maxValueTyped,
      reinterpret_cast<IN*>(minValueDevicePtr),
      numBitsDevicePtr,
      inputTyped,
      numDevice,
      maxNum);

  bitPackFixedBitAndMinInternal<IN, OUT, LIMIT>(
      minValueDevicePtr,
      numBitsDevicePtr,
      workspace,
      outPtr,
      in,
      numDevice,
      maxNum);
}


void compress(
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

  for (int n = 0; n < 100; n++)
  NVCOMP_TYPE_SWITCH(
      inType,
      bitPackInternal,
      workspace,
      outPtr,
      in,
      numPtr,
      maxNum,
      minValueDevicePtr,
      numBitsDevicePtr);
}

