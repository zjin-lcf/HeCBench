// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <sycl/sycl.hpp>

#define CUDART_INF_F  0x7f800000

namespace facebook { namespace cuda {

/// Numeric limits for CUDA
template <typename T>
struct NumericLimits {};

template<>
struct NumericLimits<float> {
  /// The minimum possible valid float (i.e., not NaN)
  inline static float minPossible() {
    return -sycl::bit_cast<float>(CUDART_INF_F);
  }

  /// The maximum possible valid float (i.e., not NaN)
  inline static float maxPossible() {
    return sycl::bit_cast<float>(CUDART_INF_F);
  }
};

template<>
struct NumericLimits<int> {
  /// The minimum possible int
  inline static int minPossible() {
    return INT_MIN;
  }

  /// The maximum possible int
  inline static int maxPossible() {
    return INT_MAX;
  }
};

template<>
struct NumericLimits<unsigned int> {
  /// The minimum possible unsigned int
  inline static unsigned int minPossible() {
    return 0;
  }

  /// The maximum possible unsigned int
  inline static unsigned int maxPossible() {
    return UINT_MAX;
  }
};

} } // namespace
