// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <sycl/sycl.hpp>

namespace facebook { namespace cuda {

/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
  inline Pair() {}

  inline Pair(K key, V value) : k(key), v(value) {}

  inline bool operator==(const Pair<K, V> &rhs) const {
    return (k == rhs.k) && (v == rhs.v);
  }

  inline bool operator!=(const Pair<K, V> &rhs) const {
    return !operator==(rhs);
  }

  inline bool operator<(const Pair<K, V> &rhs) const {
    return (k < rhs.k) || ((k == rhs.k) && (v < rhs.v));
  }

  inline bool operator>(const Pair<K, V> &rhs) const {
    return (k > rhs.k) || ((k == rhs.k) && (v > rhs.v));
  }

  K k;
  V v;
};

} } // namespace
