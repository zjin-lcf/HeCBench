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

inline float4 convert_float4(uchar4 data) 
{
   float4 r;
   r.x() = (float)data.x();
   r.y() = (float)data.y();
   r.z() = (float)data.z();
   r.w() = (float)data.w();
   return r;
}

inline uchar4 convert_uchar4(float4 v) {
  uchar4 r;
  r.x() = (unsigned char) ((v.x() > 255.f) ? 255.f : (v.x() < 0.f ? 0.f : v.x()));
  r.y() = (unsigned char) ((v.y() > 255.f) ? 255.f : (v.y() < 0.f ? 0.f : v.y()));
  r.z() = (unsigned char) ((v.z() > 255.f) ? 255.f : (v.z() < 0.f ? 0.f : v.z()));
  r.w() = (unsigned char) ((v.w() > 255.f) ? 255.f : (v.w() < 0.f ? 0.f : v.w()));
  return r;
}

void sobel_filter(const uchar4*__restrict inputImage, 
                        uchar4*__restrict outputImage, 
                  const uint width,
                  const uint height,
                  sycl::nd_item<2> &item)
{
  uint x = item.get_global_id(1);
  uint y = item.get_global_id(0);

  /* Read each texel component and calculate the filtered value using neighbouring texel components */
  if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
  {
    int c = x + y * width;

    float4 i00 = convert_float4(inputImage[c - 1 - width]);
    float4 i01 = convert_float4(inputImage[c - width]);
    float4 i02 = convert_float4(inputImage[c + 1 - width]);

    float4 i10 = convert_float4(inputImage[c - 1]);
    float4 i12 = convert_float4(inputImage[c + 1]);

    float4 i20 = convert_float4(inputImage[c - 1 + width]);
    float4 i21 = convert_float4(inputImage[c + width]);
    float4 i22 = convert_float4(inputImage[c + 1 + width]);
    
    const float4 two = (float4)(2.f);

    float4 Gx = i00 + two * i10 + i20 - i02 - two * i12 - i22;
    float4 Gy = i00 - i20  + two * i01 - two * i21 + i02 - i22;

    /* taking root of sums of squares of Gx and Gy */
    outputImage[c] = convert_uchar4(sycl::sqrt(Gx*Gx + Gy*Gy)/(float4)(2));
  }
}
