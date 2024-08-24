#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with a given displacement field, CUDA kernel.
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[in]  u       horizontal displacement
/// \param[in]  v       vertical displacement
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void WarpingKernel(int width, int height, int stride, const float *u,
                   const float *v, float *out,
                   sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                            sycl::access::target::image>
                       texToWarp,
                   sycl::sampler texDesc,
                   const sycl::nd_item<3> &item) {
  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float x = ((float)ix + u[pos]);
  float y = ((float)iy + v[pos]);

  auto inputCoord = sycl::float2(x, y);

  out[pos] = texToWarp.read(inputCoord, texDesc)[0];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with provided vector field, CUDA kernel wrapper.
///
/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
/// \param[in]  src source image
/// \param[in]  w   width
/// \param[in]  h   height
/// \param[in]  s   stride
/// \param[in]  u   horizontal displacement
/// \param[in]  v   vertical displacement
/// \param[out] out warped image
///////////////////////////////////////////////////////////////////////////////
static void WarpImage(const float *src, float *pI0_h, float *I0_h, float *src_p, int w, int h, int s, const float *u,
                      const float *v, float *out, sycl::queue &q) {
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  int dataSize = s * h * sizeof(float);
  q.memcpy(I0_h, src, dataSize).wait();

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int index = i * s + j;
      pI0_h[index * 4 + 0] = I0_h[index];
      pI0_h[index * 4 + 1] = pI0_h[index * 4 + 2] = pI0_h[index * 4 + 3] = 0.f;
    }
  }
  q.memcpy(src_p, pI0_h, s * h * sizeof(sycl::float4)).wait();

  auto texDescr = sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::linear);

  auto texToWarp =
      sycl::image<2>(src_p, sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32, sycl::range<2>(w, h),
                         sycl::range<1>(s * sizeof(sycl::float4)));
  
  q.submit([&](sycl::handler &cgh) {
    auto texToWarp_acc =
         texToWarp.template get_access<sycl::float4,
                                       sycl::access::mode::read>(cgh);

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item) {
                       WarpingKernel(w, h, s, u, v, out,
                                     texToWarp_acc, texDescr, item);
                     });
  });
}
