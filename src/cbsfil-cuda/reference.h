#ifndef CUBIC_BSPLINE_PREFILTER_CPU_H
#define CUBIC_BSPLINE_PREFILTER_CPU_H

#include <cstdint>

#define POLE_REF (std::sqrt(3.0f) - 2.0f)

typedef unsigned int  uint;
typedef unsigned char uchar;

static float InitialCausalCoefficient_ref(float* c, uint DataLength, int step)
{
  const uint Horizon = (12 < DataLength) ? 12 : DataLength;

  float zn = POLE_REF;
  float Sum = *c;

  for (uint n = 0; n < Horizon; n++) {
    Sum += zn * (*c);
    zn *= POLE_REF;
    c = reinterpret_cast<float*>(
        reinterpret_cast<uchar*>(c) + step);
  }
  return Sum;
}

static float InitialAntiCausalCoefficient_ref(float* c, uint DataLength, int step)
{
  (void)DataLength; // unused
  (void)step;       // unused
  return (POLE_REF / (POLE_REF - 1.0f)) * (*c);
}

static void ConvertToInterpolationCoefficients_ref(
    float* coeffs,
    uint DataLength,
    int step)
{
  const float Lambda = (1.0f - POLE_REF) * (1.0f - 1.0f / POLE_REF);

  // ---- Causal pass ----
  float* c = coeffs;
  float previous_c;

  *c = previous_c =
    Lambda * InitialCausalCoefficient_ref(c, DataLength, step);

  for (uint n = 1; n < DataLength; n++) {
    c = reinterpret_cast<float*>(
        reinterpret_cast<uchar*>(c) + step);

    *c = previous_c = Lambda * (*c) + POLE_REF * previous_c;
  }

  // ---- Anti-causal pass ----
  *c = previous_c =
    InitialAntiCausalCoefficient_ref(c, DataLength, step);

  for (int n = DataLength - 2; n >= 0; n--) {
    c = reinterpret_cast<float*>(
        reinterpret_cast<uchar*>(c) - step);

    *c = previous_c = POLE_REF * (previous_c - (*c));
  }
}

static void toCoef2DX_ref(
    float* image,
    uint pitch,     // pitch in bytes
    uint width,
    uint height)
{
  for (uint y = 0; y < height; y++) {
    float* line = reinterpret_cast<float*>(
        reinterpret_cast<uchar*>(image) + y * pitch);

    ConvertToInterpolationCoefficients_ref(
        line,
        width,
        sizeof(float));
  }
}

static void toCoef2DY_ref(
    float* image,
    uint pitch,     // pitch in bytes
    uint width,
    uint height)
{
  for (uint x = 0; x < width; x++) {
    float* line = image + x;
    ConvertToInterpolationCoefficients_ref(line, height, pitch);
  }
}

#endif
