//
// Copyright 2004-present Facebook. All Rights Reserved.
//

#include <sycl/sycl.hpp>
#include "cuda/CudaUtils.dp.hpp"
#include "cuda/DeviceTensor.dp.hpp"
#include "cuda/NumericLimits.dp.hpp"
#include "cuda/RegisterUtils.dp.hpp"
#include "cuda/ShuffleTypes.dp.hpp"
#include "cuda/WarpBitonicSort.dp.hpp"
#include <assert.h>

namespace facebook { namespace cuda {

/** @file

    Sorting routines for in-register warp shuffle sorting. Can
    handle arbitrary sizes via recursive decomposition as long as you
    have enough registers allocated for the problem, though efficiency
    dies as you get significantly larger than 128.

    Template instantiations are provided for input sizes 1-128 (`4 * warpSize`).
*/

// Retrieves 16 values from the warp-held arr[N] from `index` to `index
// + 15`, considering it as an index across the warp threads.
// e.g., returns arr[index / warpSize] from lanes
// L = 16 * ((index / halfWarpSize) % 2) to L + 15 inclusive.
template <int N, typename T>
T
getMulti(const T arr[N], int index, T val, sycl::nd_item<1> &item) {
  // Only lanes 0-15 load new data
  const int bucket = index / WARP_SIZE;

  // 0 if we are reading using lanes 0-15, 1 otherwise
  const int halfWarp = (index / HALF_WARP_SIZE) & 0x1;

  T load = RegisterIndexUtils<T, N>::get(arr, bucket);
  T shift = shfl_down(load, HALF_WARP_SIZE, item);

  load = halfWarp ? shift : load;
  return (getLaneId(item) < HALF_WARP_SIZE) ? load : val;
}

// Writes 16 values from the warp-held arr[N] from `index` to `index +
// 15`, considering it as an index across the warp threads. Only
// writes `val` from the first half-warp.
// e.g, sets arr[index / warpSize] in lanes
// L = 16 * ((index / halfWarpSize) % 2) to L + 15 inclusive, using
// the values in `val` from lanes 0-15 inclusive.
template <int N, typename T>
void
scatterHalfWarp(T arr[N], int index, T val, sycl::nd_item<1> &item) {
  // Destination index of arr
  const int bucket = index / WARP_SIZE;

  // Which half warp threads participate in the write?
  // 0 if arr[bucket]:lane 0-15 = val:lane 0-15
  // 1 if arr[bucket]:lane 16-31 = val:lane 0-15
  // `val` always comes from lanes 0-15
  const int halfWarp = (index / HALF_WARP_SIZE) & 0x1;

  // If we are writing to lanes 16-31, we need to get the value from
  // lanes 0-15
  T shift = shfl_up(val, HALF_WARP_SIZE, item);
  val = halfWarp ? shift : val;

  // Are we in the half-warp that we want to be in?
  if ((getLaneId(item) & (halfWarp << 4)) == (halfWarp << 4)) {
    RegisterIndexUtils<T, N>::set(arr, bucket, val);
  }
}

// Performs the merging step of a merge sort within a warp, using
// registers `a[M]` and `b[N]` as the two sorted input lists,
// outputting a sorted `dst[M + N]`. All storage is in registers
// across the warp, and uses warp shuffles for data manipulation.
template <typename T, typename Comparator, int M, int N>
void
warpMergeMN(const T a[M], const T b[N], T dst[M + N], sycl::nd_item<1> &item) {
  const int laneId = getLaneId(item);

  // It is presumed that `a` and `b` are sorted lists of the form:
  // A: [0:lane 0-31], [1: lane 0-31], ..., [N-1: lane 0-31]
  // B: [0:lane 0-31], [1: lane 0-31], ..., [N-1: lane 0-31]
  // We are merging `a` and `b` into dst

  // `val` is the working array that we are merging into and sorting
  // Populate `val` with initial values:
  // lanes 0-15 take a[0:lane 0-15], lanes 16-31 take b[0:lane 0-15]
  T val = shfl_up(b[0], HALF_WARP_SIZE, item);
  val = (laneId < HALF_WARP_SIZE) ? a[0] : val;

  int aIndex = HALF_WARP_SIZE;
  int bIndex = HALF_WARP_SIZE;
  int dstIndex = 0;

  // Each time through we use a sort of 32 elements as our merge
  // primitive, and output 16 elements. For the following 16 elements,
  // we take from either `a` or `b` depending on which list is
  // guaranteed to have the larger or equivalent elements to the
  // following list.
  for ( ; ; ) {
    // Sort entries in `val`
    val = warpBitonicSort<T, Comparator>(val, item);

    if (dstIndex < ((M + N) * WARP_SIZE - WARP_SIZE)) {
      // Values val[lane 0-15] are sorted, output them
      scatterHalfWarp<M + N>(dst, dstIndex, val, item);
      dstIndex += HALF_WARP_SIZE;
    } else {
      // We've exhausted `a` and `b`. Everything left in `val` across
      // all lanes are the final values
      assert(aIndex == WARP_SIZE * M);
      assert(bIndex == WARP_SIZE * N);
      dst[M + N - 1] = val;
      break;
    }

    // It is possible that we've exhausted one of the branches (A or
    // B).
    if (aIndex == WARP_SIZE * M) {
      // We have to load from `b`; `a` has no more elements
      val = getMulti<N>(b, bIndex, val, item);
      bIndex += HALF_WARP_SIZE;
    } else if (bIndex == WARP_SIZE * N) {
      // We have to load from `a`; `b` has no more elements
      val = getMulti<M>(a, aIndex, val, item);
      aIndex += HALF_WARP_SIZE;
    } else {
      // Should we take from `a` or `b` next?
      const T compA =
          WarpRegisterUtils<T, M>::broadcast(a, aIndex - 1, item);
      const T compB =
          WarpRegisterUtils<T, N>::broadcast(b, bIndex - 1, item);

      if (Comparator::compare(compA, compB)) {
        // Load from `a` next
        val = getMulti<M>(a, aIndex, val, item);
        aIndex += HALF_WARP_SIZE;
      } else {
        // Load from `b` next
        val = getMulti<N>(b, bIndex, val, item);
        bIndex += HALF_WARP_SIZE;
      }
    }
  }
}

#define STATIC_FLOOR(N, DIV) (int) (N / DIV)
#define STATIC_CEIL(N, DIV) (int) ((N + DIV - 1) / DIV)

// Recursive merging of N sorted lists into 1 sorted list
template <typename T, typename Comparator, int N>
struct Merge {
  static void splitAndMerge(const T in[N], T out[N], sycl::nd_item<1> &item) {
    // Split the input into two sub-lists `a` and `b` as best as
    // possible
    T a[STATIC_FLOOR(N, 2)];
    T b[STATIC_CEIL(N, 2)];

    for (int i = 0; i < STATIC_FLOOR(N, 2); ++i) {
      a[i] = in[i];
    }

    for (int i = STATIC_FLOOR(N, 2); i < N; ++i) {
      b[i - STATIC_FLOOR(N, 2)] = in[i];
    }

    // Recursively split `a` and merge to a sorted list `aOut`
    T aOut[STATIC_FLOOR(N, 2)];
    Merge<T, Comparator, STATIC_FLOOR(N, 2)>::splitAndMerge(a, aOut, item);

    // Recursively split `b` and merge to a sorted list `bOut`
    T bOut[STATIC_CEIL(N, 2)];
    Merge<T, Comparator, STATIC_CEIL(N, 2)>::splitAndMerge(b, bOut, item);

    // Merge `aOut` with `bOut` to produce the final sorted list `out`
    warpMergeMN<T, Comparator, STATIC_FLOOR(N, 2), STATIC_CEIL(N, 2)>(
        aOut, bOut, out, item);
  }
};

#undef STATIC_FLOOR
#undef STATIC_CEIL

// Base case: 1 list requires no merging
template <typename T, typename Comparator>
struct Merge<T, Comparator, 1> {
  static void splitAndMerge(const T in[1], T out[1], sycl::nd_item<1> &item) {
    out[0] = in[0];
  }
};

template <typename T, typename Comparator, int N>
void warpSortRegisters(T a[N], T dst[N], sycl::nd_item<1> &item) {
  // Sort all sub-lists of 32. We could do this in Merge's base case
  // instead, but that increases register usage, since it is at the
  // leaf of the recursion.
  for (int i = 0; i < N; ++i) {
    a[i] = warpBitonicSort<T, Comparator>(a[i], item);
  }

  // Recursive subdivision to sort all a[i] together
  Merge<T, Comparator, N>::splitAndMerge(a, dst, item);
}

// Sort keys only
template <typename T, typename Comparator, int N>
void
warpSortRegisters(const DeviceTensor<T, 1>& key,
                  DeviceTensor<T, 1>& sortedKey,
                  sycl::nd_item<1> &item) {

  // Load the elements we have available
  T val[N];
  WarpRegisterLoaderUtils<T, N>::load(val, key, NumericLimits<T>::minPossible(), item);

  // Recursively split, shuffle sort and merge sort back
  T sortedVal[N];
  warpSortRegisters<T, Comparator, N>(val, sortedVal, item);

  // Write the warp's registers back out
  WarpRegisterLoaderUtils<T, N>::save(sortedKey, sortedVal, key.getSize(0),
                                      item);
}

// Sort keys, writing the sorted keys and the original indices of the
// sorted keys out into two different arrays
template <typename T, typename IndexType, typename Comparator, int N>
void
warpSortRegisters(const DeviceTensor<T, 1>& key,
                  DeviceTensor<T, 1>& sortedKey,
                  DeviceTensor<IndexType, 1>& sortedKeyIndices,
                  sycl::nd_item<1> &item) {

  // Load the elements we have available
  Pair<T, int> pairs[N];
  WarpRegisterPairLoaderUtils<T, IndexType, N>::load(
      pairs, key, NumericLimits<T>::minPossible(),
      NumericLimits<IndexType>::minPossible(), item);

  // Recursively split, shuffle sort and merge sort back
  Pair<T, IndexType> sortedPairs[N];
  warpSortRegisters<Pair<T, IndexType>, Comparator, N>(pairs, sortedPairs, item);

  // Write the warp's registers back out
  WarpRegisterPairLoaderUtils<T, IndexType, N>::save(
      sortedKey, sortedKeyIndices, sortedPairs, key.getSize(0), item);
}

// Sort a key/value pair in two different arrays
template <typename K, typename V, typename Comparator, int N>
void
warpSortRegisters(const DeviceTensor<K, 1>& key,
                  const DeviceTensor<V, 1>& value,
                  DeviceTensor<K, 1>& sortedKey,
                  DeviceTensor<V, 1>& sortedValue,
                  sycl::nd_item<1> &item) {

  // Load the elements we have available
  Pair<K, V> pairs[N];
  WarpRegisterPairLoaderUtils<K, V, N>::load(
    pairs, key, value,
    NumericLimits<K>::minPossible(),
    NumericLimits<V>::minPossible());

  // Recursively split, shuffle sort and merge sort back
  Pair<K, V> sortedPairs[N];
  warpSortRegisters<Pair<K, V>, Comparator, N>(pairs, sortedPairs, item);

  // Write the warp's registers back out
  WarpRegisterPairLoaderUtils<K, V, N>::save(
    sortedKey, sortedValue, sortedPairs, key.getSize(0));
}

// Sort keys only; returns true if we could handle an array of this size
template <typename T, typename Comparator>
bool warpSort(const DeviceTensor<T, 1>& key,
                         DeviceTensor<T, 1>& sortedKey,
                         sycl::nd_item<1> &item) {
  assert(key.getSize(0) <= sortedKey.getSize(0));

  if (key.getSize(0) <= WARP_SIZE) {
    warpSortRegisters<float, Comparator, 1>(key, sortedKey, item);
    return true;
  } else if (key.getSize(0) <= 2 * WARP_SIZE) {
    warpSortRegisters<float, Comparator, 2>(key, sortedKey, item);
    return true;
  } else if (key.getSize(0) <= 3 * WARP_SIZE) {
    warpSortRegisters<float, Comparator, 3>(key, sortedKey, item);
    return true;
  } else if (key.getSize(0) <= 4 * WARP_SIZE) {
    warpSortRegisters<float, Comparator, 4>(key, sortedKey, item);
    return true;
  }

  // size too large
  return false;
}

// Sort keys, writing the sorted keys and the original indices of the
// sorted keys out into two different arrays. Returns true if we could
// handle an array of this size.
template <typename T, typename IndexType, typename Comparator>
bool warpSort(const DeviceTensor<T, 1>& key,
                         DeviceTensor<T, 1>& sortedKey,
                         DeviceTensor<IndexType, 1>& sortedKeyIndices,
                         sycl::nd_item<1> &item) {
  assert(key.getSize(0) <= sortedKey.getSize(0) &&
         key.getSize(0) <= sortedKeyIndices.getSize(0));

  if (key.getSize(0) <= WARP_SIZE) {
    warpSortRegisters<float, IndexType, Comparator, 1>(
      key, sortedKey, sortedKeyIndices, item);
    return true;
  } else if (key.getSize(0) <= 2 * WARP_SIZE) {
    warpSortRegisters<float, IndexType, Comparator, 2>(
      key, sortedKey, sortedKeyIndices, item);
    return true;
  } else if (key.getSize(0) <= 3 * WARP_SIZE) {
    warpSortRegisters<float, IndexType, Comparator, 3>(
      key, sortedKey, sortedKeyIndices, item);
    return true;
  } else if (key.getSize(0) <= 4 * WARP_SIZE) {
    warpSortRegisters<float, IndexType, Comparator, 4>(
      key, sortedKey, sortedKeyIndices, item);
    return true;
  }

  // size too large
  return false;
}

// Sort a key/value pair in two different arrays. Returns true if we
// could handle an array of this size.
template <typename K, typename V, typename Comparator>
bool warpSort(const DeviceTensor<K, 1>& key,
                         const DeviceTensor<V, 1>& value,
                         DeviceTensor<K, 1>& sortedKey,
                         DeviceTensor<V, 1>& sortedValue,
                         sycl::nd_item<1> &item) {
  assert(key.getSize(0) <= sortedKey.getSize(0) &&
         value.getSize(0) <= sortedValue.getSize(0) &&
         key.getSize(0) == value.getSize(0));

  if (key.getSize(0) <= WARP_SIZE) {
    warpSortRegisters<K, V, Comparator, 1>(key, value, sortedKey, sortedValue,
                                           item);
    return true;
  } else if (key.getSize(0) <= 2 * WARP_SIZE) {
    warpSortRegisters<K, V, Comparator, 2>(key, value, sortedKey, sortedValue,
                                           item);
    return true;
  } else if (key.getSize(0) <= 3 * WARP_SIZE) {
    warpSortRegisters<K, V, Comparator, 3>(key, value, sortedKey, sortedValue,
                                           item);
    return true;
  } else if (key.getSize(0) <= 4 * WARP_SIZE) {
    warpSortRegisters<K, V, Comparator, 4>(key, value, sortedKey, sortedValue,
                                           item);
    return true;
  }

  // size too large
  return false;
}

// Define sort kernels

void
sortDevice(DeviceTensor<float, 1> data, DeviceTensor<float, 1> out,
           sycl::nd_item<1> &item) {
  warpSort<float, GreaterThan<float>>(data, out, item);
}

void
sortDevice(DeviceTensor<float, 1> data,
           DeviceTensor<float, 1> out,
           DeviceTensor<int, 1> indices,
           sycl::nd_item<1> &item) {
  warpSort<float, int, GreaterThan<Pair<float, int>>>(data, out, indices, item);
}

// Define sort functions called in the main

std::vector<float> sort(sycl::queue &q,
                        const std::vector<float> &data,
                        double &time) {
  const size_t sizeBytes = data.size() * sizeof(float);

  float* devFloat = NULL;
  devFloat = (float *)sycl::malloc_device(sizeBytes, q);
  q.memcpy(devFloat, data.data(), sizeBytes).wait();

  float* devResult = NULL;
  devResult = (float *)sycl::malloc_device(sizeBytes, q);
  q.memset(devResult, 0, sizeBytes).wait();

  sycl::range<1> gws (32);
  sycl::range<1> lws (32);

  int dataSizes[] = { (int) data.size() };
  int outSizes[] = { (int) data.size() };

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.parallel_for(
    sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
      sortDevice(DeviceTensor<float, 1>(devFloat, dataSizes),
                 DeviceTensor<float, 1>(devResult, outSizes), item);
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::vector<float> vals(data.size());
  q.memcpy(vals.data(), devResult, sizeBytes).wait();

  sycl::free(devFloat, q);
  sycl::free(devResult, q);

  return vals;
}

std::vector<std::pair<float, int>>
sortWithIndices(sycl::queue &q,
                const std::vector<float> &data,
                double &time) {

  const size_t sizeBytes = data.size() * sizeof(float);
  const size_t sizeIndicesBytes = data.size() * sizeof(int);

  float* devFloat = NULL;
  devFloat = (float *)sycl::malloc_device(sizeBytes, q);
  q.memcpy(devFloat, data.data(), sizeBytes).wait();

  float* devResult = NULL;
  devResult = (float *)sycl::malloc_device(sizeBytes, q);
  q.memset(devResult, 0, sizeBytes).wait();

  int* devIndices = NULL;
  devIndices = (int *)sycl::malloc_device(sizeIndicesBytes, q);
  q.memset(devIndices, 0, sizeIndicesBytes).wait();

  sycl::range<1> gws (32);
  sycl::range<1> lws (32);

  int dataSizes[] = { (int) data.size() };
  int outSizes[] = { (int) data.size() };

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.parallel_for(
    sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
      sortDevice(DeviceTensor<float, 1>(devFloat, dataSizes),
                 DeviceTensor<float, 1>(devResult, outSizes),
                 DeviceTensor<int, 1>(devIndices, outSizes), item);
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::vector<float> vals(data.size());
  q.memcpy(vals.data(), devResult, sizeBytes).wait();

  std::vector<int> indices(data.size());
  q.memcpy(indices.data(), devIndices, sizeIndicesBytes).wait();

  sycl::free(devFloat, q);
  sycl::free(devResult, q);
  sycl::free(devIndices, q);

  std::vector<std::pair<float, int> > result;
  for (int i = 0; i < (int)data.size(); ++i) {
    result.push_back(std::make_pair(vals[i], indices[i]));
  }

  return result;
}

} } // namespace
