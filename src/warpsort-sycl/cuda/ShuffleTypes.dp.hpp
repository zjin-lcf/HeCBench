// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <sycl/sycl.hpp>
#include "cuda/Pair.dp.hpp"

namespace facebook { namespace cuda {

/** @file

    Templated warp shuffles that work for basic and pair types
*/

template <typename T>
inline T shfl(const T val, int srcLane, sycl::nd_item<1> &item,
                       int width = WARP_SIZE) {
  return sycl::select_from_group(item.get_sub_group(), val, srcLane);
}

template <typename T>
inline T shfl_up(const T val, int delta, sycl::nd_item<1> &item,
                          int width = WARP_SIZE) {
  return sycl::shift_group_right(item.get_sub_group(), val, delta);
}

template <typename T>
inline T shfl_down(const T val, int delta, sycl::nd_item<1> &item,
                            int width = WARP_SIZE) {
  return sycl::shift_group_left(item.get_sub_group(), val, delta);
}

template <typename T>
inline T shfl_xor(const T val, int laneMask, sycl::nd_item<1> &item,
                           int width = WARP_SIZE) {
  return sycl::permute_group_by_xor(item.get_sub_group(), val, laneMask);
}

template <typename K, typename V>
inline Pair<K, V> shfl(const Pair<K, V> &p, int srcLane,
                                sycl::nd_item<1> &item,
                                int width = WARP_SIZE) {
  return Pair<K, V>(
      sycl::select_from_group(item.get_sub_group(), p.k, srcLane),
      sycl::select_from_group(item.get_sub_group(), p.v, srcLane));
}

template <typename K, typename V>
inline Pair<K, V> shfl_up(const Pair<K, V> &p, int delta,
                                   sycl::nd_item<1> &item,
                                   int width = WARP_SIZE) {
  return Pair<K, V>(
      sycl::shift_group_right(item.get_sub_group(), p.k, delta),
      sycl::shift_group_right(item.get_sub_group(), p.v, delta));
}

template <typename K, typename V>
inline Pair<K, V> shfl_down(const Pair<K, V> &p, int delta,
                                     sycl::nd_item<1> &item,
                                     int width = WARP_SIZE) {
  return Pair<K, V>(
      sycl::shift_group_left(item.get_sub_group(), p.k, delta),
      sycl::shift_group_left(item.get_sub_group(), p.v, delta));
}

template <typename K, typename V>
inline Pair<K, V> shfl_xor(const Pair<K, V> &p, int laneMask,
                                    sycl::nd_item<1> &item,
                                    int width = WARP_SIZE) {
  return Pair<K, V>(
      sycl::permute_group_by_xor(item.get_sub_group(), p.k, laneMask),
      sycl::permute_group_by_xor(item.get_sub_group(), p.v, laneMask));
}

} } // namespace
