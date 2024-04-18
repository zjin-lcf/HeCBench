#ifndef UTILS_H
#define UTILS_H

#include <xpu/Macros.h>
#include <xpu/Stream.h>
#include <c10/core/Device.h>

inline sycl::queue& getCurrentXPUQueue() 
{
  auto device_type = c10::DeviceType::XPU; 
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream xpu_stream = impl.getStream(impl.getDevice());
  return xpu::get_queue_from_stream(xpu_stream);
}

#endif

