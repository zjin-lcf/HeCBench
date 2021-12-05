
#ifndef _CUBIC_BSPLINE_PREFILTER_KERNEL_H_
#define _CUBIC_BSPLINE_PREFILTER_KERNEL_H_

// The code below is based on the work of Philippe Thevenaz.
// See <http://bigwww.epfl.ch/thevenaz/interpolation/>

#define POLE (sqrtf(3.0f)-2.0f)  //pole for cubic b-spline

typedef unsigned int uint;
typedef unsigned char uchar;

#pragma omp declare target
float InitialCausalCoefficient(
    float* c,         // coefficients
    uint DataLength,  // number of coefficients
    int step)         // element interleave in bytes
{
  const uint Horizon = 12 < DataLength ? 12 : DataLength;

  // this initialization corresponds to clamping boundaries
  // accelerated loop
  float zn = POLE;
  float Sum = *c;
  for (uint n = 0; n < Horizon; n++) {
    Sum += zn * *c;
    zn *= POLE;
    c = (float*)((uchar*)c + step);
  }
  return(Sum);
}

float InitialAntiCausalCoefficient(
    float* c,         // last coefficient
    uint DataLength,  // number of samples or coefficients
    int step)         // element interleave in bytes
{
  // this initialization corresponds to clamping boundaries
  return((POLE / (POLE - 1.0f)) * *c);
}

void ConvertToInterpolationCoefficients(
    float* coeffs,    // input samples --> output coefficients
    uint DataLength,  // number of samples or coefficients
    int step)         // element interleave in bytes
{
  // compute the overall gain
  const float Lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

  // causal initialization
  float* c = coeffs;
  float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
  *c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
  // causal recursion
  for (uint n = 1; n < DataLength; n++) {
    c = (float*)((uchar*)c + step);
    *c = previous_c = Lambda * *c + POLE * previous_c;
  }
  // anticausal initialization
  *c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
  // anticausal recursion
  for (int n = DataLength - 2; 0 <= n; n--) {
    c = (float*)((uchar*)c - step);
    *c = previous_c = POLE * (previous_c - *c);
  }
}
#pragma omp end declare target

void toCoef2DX(
    float* image,
    uint numThreads,
    uint pitch,
    uint width,
    uint height)
{
  // process lines horizontally
  #pragma omp target teams distribute parallel for thread_limit(numThreads)
  for (uint y = 0; y < height; y++) {
    float* line = (float*)((uchar*)image + y * pitch);  //direct access
    ConvertToInterpolationCoefficients(line, width, sizeof(float));
  }
}

void toCoef2DY(
    float* image,
    uint numThreads,
    uint pitch,
    uint width,
    uint height)
{
  // process lines vertically
  #pragma omp target teams distribute parallel for thread_limit(numThreads)
  for (uint x = 0; x < width; x++) {
    float* line = image + x;  //direct access
    ConvertToInterpolationCoefficients(line, height, pitch);
  }
}

#endif // _CUBIC_BSPLINE_PREFILTER_KERNEL_H_
