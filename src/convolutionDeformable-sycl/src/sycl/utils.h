#ifndef UTILS_H
#define UTILS_H

#include <c10/xpu/XPUStream.h>

inline sycl::queue& getCurrentXPUQueue() 
{
  return c10::xpu::getCurrentXPUStream().queue();
}

#endif

