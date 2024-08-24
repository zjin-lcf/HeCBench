#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, CUDA kernel
/// \param[in]  width   field width
/// \param[in]  height  field height
/// \param[in]  stride  field stride
/// \param[in]  scale   scale factor (multiplier)
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void UpscaleKernel(int width, int height, int stride, float scale, float *out,
                  sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                            sycl::access::target::image> texCoarse_acc,
                   sycl::sampler texDesc,
                   const sycl::nd_item<3> &item) {
  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);

  if (ix >= width || iy >= height) return;

  float x = ((float)ix - 0.5f) * 0.5f;
  float y = ((float)iy - 0.5f) * 0.5f;

  auto inputCoord = sycl::float2(x, y);

  // exploit hardware interpolation
  // and scale interpolated vector to match next pyramid level resolution
  out[ix + iy * stride] = texCoarse_acc.read(inputCoord, texDesc)[0] * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static void Upscale(const float *src, float *pI0_h, float *I0_h, float *src_p, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride, float scale,
                    float *out, sycl::queue &q) {
  sycl::range<3> threads(1, 8, 32);
  sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                        iDivUp(newWidth, threads[2]));

  int dataSize = stride * height * sizeof(float);
  q.memcpy(I0_h, src, dataSize).wait();

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * stride + j;
      pI0_h[index * 4 + 0] = I0_h[index];
      pI0_h[index * 4 + 1] = pI0_h[index * 4 + 2] = pI0_h[index * 4 + 3] = 0.f;
    }
  }
  q.memcpy(src_p, pI0_h, height * stride * sizeof(sycl::float4)).wait();

  auto texDescr = sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::linear);

  auto texCoarse = sycl::image<2>(
      src_p, sycl::image_channel_order::rgba,
      sycl::image_channel_type::fp32, sycl::range<2>(width, height),
      sycl::range<1>(stride * sizeof(sycl::float4)));
  
  q.submit([&](sycl::handler &cgh) {
    auto texCoarse_acc =
         texCoarse.template get_access<sycl::float4,
                                       sycl::access::mode::read>(cgh);

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item) {
                       UpscaleKernel(newWidth, newHeight, newStride, scale, out,
                                     texCoarse_acc, texDescr, item);
                     });
  });
}
