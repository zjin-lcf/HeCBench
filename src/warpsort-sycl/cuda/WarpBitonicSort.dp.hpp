// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <sycl/sycl.hpp>
#include "cuda/Comparators.dp.hpp"
#include "cuda/CudaUtils.dp.hpp"
#include "cuda/ShuffleTypes.dp.hpp"

namespace facebook { namespace cuda {

namespace detail {

template <typename T, typename Comparator>
inline T shflSwap(const T x, int mask, int dir,
                           sycl::nd_item<1> &item) {
  T y = shfl_xor(x, mask, item);
  return Comparator::compare(x, y) == dir ? y : x;
}

} // namespace

/// Defines a bitonic sort network to exchange 'V' according to
/// `SWAP()`'s compare and exchange mechanism across the warp, ordered
/// according to the comparator `comp`. In other words, if `comp` is
/// `GreaterThan<T>`, then lane 0 will contain the highest `val`
/// presented across the warp
///
/// See also 
/// http://on-demand.gputechconf.com/gtc/2013/presentations/S3174-Kepler-Shuffle-Tips-Tricks.pdf
template <typename T, typename Comparator>
T warpBitonicSort(T val, sycl::nd_item<1> &item) {
  const int laneId = getLaneId(item);
  // 2
  val = detail::shflSwap<T, Comparator>(
      val, 0x01, getBit(laneId, 1) ^ getBit(laneId, 0), item);

  // 4
  val = detail::shflSwap<T, Comparator>(
      val, 0x02, getBit(laneId, 2) ^ getBit(laneId, 1), item);
  val = detail::shflSwap<T, Comparator>(
      val, 0x01, getBit(laneId, 2) ^ getBit(laneId, 0), item);

  // 8
  val = detail::shflSwap<T, Comparator>(
      val, 0x04, getBit(laneId, 3) ^ getBit(laneId, 2), item);
  val = detail::shflSwap<T, Comparator>(
      val, 0x02, getBit(laneId, 3) ^ getBit(laneId, 1), item);
  val = detail::shflSwap<T, Comparator>(
      val, 0x01, getBit(laneId, 3) ^ getBit(laneId, 0), item);

  // 16
  val = detail::shflSwap<T, Comparator>(
      val, 0x08, getBit(laneId, 4) ^ getBit(laneId, 3), item);
  val = detail::shflSwap<T, Comparator>(
      val, 0x04, getBit(laneId, 4) ^ getBit(laneId, 2), item);
  val = detail::shflSwap<T, Comparator>(
      val, 0x02, getBit(laneId, 4) ^ getBit(laneId, 1), item);
  val = detail::shflSwap<T, Comparator>(
      val, 0x01, getBit(laneId, 4) ^ getBit(laneId, 0), item);

  // 32
  val = detail::shflSwap<T, Comparator>(val, 0x10, getBit(laneId, 4), item);
  val = detail::shflSwap<T, Comparator>(val, 0x08, getBit(laneId, 3), item);
  val = detail::shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 2), item);
  val = detail::shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 1), item);
  val = detail::shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 0), item);

  return val;
}

} } // namespace
