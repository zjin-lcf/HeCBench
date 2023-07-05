// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <sycl/sycl.hpp>

namespace facebook { namespace cuda {

/**
   Computes ceil(a / b)
*/
template <typename T> inline constexpr T ceil(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes floor(a / b)
*/
template <typename T> inline constexpr T floor(T a, T b) {
  return (a - b + 1) / b;
}

/**
   Returns the current thread's warp ID
*/
inline int getWarpId(sycl::nd_item<1> &item) {
  return item.get_local_id(0) / WARP_SIZE;
}

/**
   Returns the number of threads in the current block (linearized).
*/
inline int getThreadsInBlock(sycl::nd_item<1> &item) {
  return item.get_local_range(0);
}

/**
   Returns the number of warps in the current block (linearized,
   rounded to whole warps).
*/
inline int getWarpsInBlock(sycl::nd_item<1> &item) {
  return ceil(getThreadsInBlock(item), WARP_SIZE);
}


/**
   Return the current thread's lane in the warp
*/
inline int getLaneId(sycl::nd_item<1> &item) {
  int laneId = item.get_local_id(0) % WARP_SIZE;
  return laneId;
}

/**
   Extract a single bit at `pos` from `val`
*/

inline int getBit(int val, int pos) {
  return (val >> pos) & 0x1;
}

/**
   Returns the index of the most significant 1 bit in `val`.
*/

inline constexpr int getMSB(int val) {
  return
    ((val >= 1024 && val < 2048) ? 10 :
     ((val >= 512) ? 9 :
      ((val >= 256) ? 8 :
       ((val >= 128) ? 7 :
        ((val >= 64) ? 6 :
         ((val >= 32) ? 5 :
          ((val >= 16) ? 4 :
           ((val >= 8) ? 3 :
            ((val >= 4) ? 2 :
             ((val >= 2) ? 1 :
              ((val == 1) ? 0 : -1)))))))))));
}

} }  // namespace
