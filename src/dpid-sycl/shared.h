#ifndef _SHARED
#define _SHARED

#include <sycl/sycl.hpp>

using uchar3 = sycl::uchar3;
using float4 = sycl::float4;

struct Params {
  uint32_t oWidth;
  uint32_t oHeight;
  uint32_t iWidth;
  uint32_t iHeight;
     float pWidth;
     float pHeight;
     float lambda;
  uint32_t repeat;
};

#endif
