#ifndef DISTORTION_H
#define DISTORTION_H

#include "common.h"

struct Properties{
  float K;
  float centerX;
  float centerY;
  int width;
  int height;
  float thresh;
  float xscale;
  float yscale;
  float xshift;
  float yshift;
};

float calc_shift(float x1, float x2, float cx, float k, float thresh);

void reference (
  const uchar3* src,
        uchar3* dst,
  const struct Properties* prop);

SYCL_EXTERNAL
void barrel_distort (
  nd_item<2> &item,
  const uchar3*__restrict src,
        uchar3*__restrict dst,
  const struct Properties*__restrict prop);

#endif
