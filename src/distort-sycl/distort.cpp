#include <cmath>
#include "distort.h"

float calc_shift(float x1, float x2, float cx, float k, float thresh)
{
  float x3 = x1 + (x2 - x1) * 0.5f;
  float result1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)));
  float result3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)));

  if(result1 > -thresh and result1 < thresh)
    return x1;
  if(result3 < 0)
  {
    return calc_shift(x3, x2, cx, k, thresh);
  }
  else
  {
    return calc_shift(x1, x3, cx, k, thresh);
  }
}

inline float getRadialX(float x, float y, const struct Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  float result = x + ((x - prop->centerX) * prop->K *
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
  return result;
}

inline float getRadialY(float x, float y, const struct Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  float result = y + ((y - prop->centerY) * prop->K * 
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
  return result;
}

inline void sampleImageTest(const sycl::uchar3* src, float idx0, float idx1,
                            sycl::uchar3& result, const struct Properties* prop)
{
  // out-of-bound check
  if((idx0 < 0) || (idx1 < 0) || (idx0 > prop->height - 1) || (idx1 > prop->width - 1))
  {
    result.x() = 0;
    result.y() = 0;
    result.z() = 0;
    return;
  }

  int idx0_floor = (int)sycl::floor(idx0);
  int idx0_ceil = (int)sycl::ceil(idx0);
  int idx1_floor = (int)sycl::floor(idx1);
  int idx1_ceil = (int)sycl::ceil(idx1);

  sycl::uchar3 s1 = src[(idx0_floor * prop->width) + idx1_floor];
  sycl::uchar3 s2 = src[(idx0_floor * prop->width) + idx1_ceil];
  sycl::uchar3 s3 = src[(idx0_ceil * prop->width) + idx1_ceil];
  sycl::uchar3 s4 = src[(idx0_ceil * prop->width) + idx1_floor];

  float x = idx0 - idx0_floor;
  float y = idx1 - idx1_floor;

  result.x() = s1.x() * (1.f - x) * (1.f - y) + s2.x() * (1.f - x) * y +
               s3.x() * x * y + s4.x() * x * (1.f - y);
  result.y() = s1.y() * (1.f - x) * (1.f - y) + s2.y() * (1.f - x) * y +
               s3.y() * x * y + s4.y() * x * (1.f - y);
  result.z() = s1.z() * (1.f - x) * (1.f - y) + s2.z() * (1.f - x) * y +
               s3.z() * x * y + s4.z() * x * (1.f - y);
}

SYCL_EXTERNAL
void barrel_distort (
  sycl::nd_item<2> &item,
  const sycl::uchar3 *__restrict src,
        sycl::uchar3 *__restrict dst,
  const struct Properties *__restrict prop)
{
  int h = item.get_global_id(0);
  int w = item.get_global_id(1);
  if (w < prop->width && h < prop->height) {
    float x = getRadialX((float)w, (float)h, prop);
    float y = getRadialY((float)w, (float)h, prop);
    sycl::uchar3 temp;
    sampleImageTest(src, y, x, temp, prop);
    dst[(h * prop->width) + w] = temp;
  }
}

void reference (
  const sycl::uchar3 *src,
        sycl::uchar3 *dst,
  const struct Properties *prop)
{
  for (int h = 0; h < prop->height; h++) {
    for (int w = 0; w < prop->width; w++) {
      float x = getRadialX((float)w, (float)h, prop);
      float y = getRadialY((float)w, (float)h, prop);
      sycl::uchar3 temp;
      sampleImageTest(src, y, x, temp, prop);
      dst[(h * prop->width) + w] = temp;
    }
  }
}
