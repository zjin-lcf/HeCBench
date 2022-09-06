#ifndef DISTORTION_H
#define DISTORTION_H

#include <omp.h>

#define max(a, b) (a) < (b) ? (b) : (a)

typedef struct __attribute__((__aligned__(4)))
{
  unsigned char x, y, z;
} uchar3;

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

void barrel_distort (
  const uchar3*__restrict src,
        uchar3*__restrict dst,
  const struct Properties*__restrict prop);

#endif
