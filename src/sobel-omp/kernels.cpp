/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each thread calculates a pixel component(rgba), by applying a filter 
 * on group of 8 neighbouring pixels in both x and y directions. 
 * Both filters are summed (vector sum) to form the final result.
 */

#pragma omp declare target
inline float4 convert_float4(uchar4 data) 
{
   return {(float)data.x, (float)data.y, (float)data.z, (float)data.w};
}

inline uchar4 convert_uchar4(float4 v) {
  uchar4 res;
  res.x = (uchar) ((v.x > 255.f) ? 255.f : (v.x < 0.f ? 0.f : v.x));
  res.y = (uchar) ((v.y > 255.f) ? 255.f : (v.y < 0.f ? 0.f : v.y));
  res.z = (uchar) ((v.z > 255.f) ? 255.f : (v.z < 0.f ? 0.f : v.z));
  res.w = (uchar) ((v.w > 255.f) ? 255.f : (v.w < 0.f ? 0.f : v.w));
  return res;
}

inline float4 operator+(float4 a, float4 b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w};
}

inline float4 operator-(float4 a, float4 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w};
}

inline float4 operator*(float4 a, float4 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w};
}

#pragma omp end declare target
