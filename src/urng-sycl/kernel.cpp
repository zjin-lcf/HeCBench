/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

inline float4 convert_float4(uchar4 v) {
  float4 res;
  res.x() = (float) v.x();
  res.y() = (float) v.y();
  res.z() = (float) v.z();
  res.w() = (float) v.w();
  return res;
}

inline uchar4 convert_uchar4_sat(float4 v) {
  uchar4 res;
  res.x() = (uchar) ((v.x() > 255.f) ? 255.f : (v.x() < 0.f ? 0.f : v.x()));
  res.y() = (uchar) ((v.y() > 255.f) ? 255.f : (v.y() < 0.f ? 0.f : v.y()));
  res.z() = (uchar) ((v.z() > 255.f) ? 255.f : (v.z() < 0.f ? 0.f : v.z()));
  res.w() = (uchar) ((v.w() > 255.f) ? 255.f : (v.w() < 0.f ? 0.f : v.w()));
  return res;
}

/* Generate uniform random deviation */
/* Park-Miller with Bays-Durham shuffle and added safeguards
   Returns a uniform random deviate between (-FACTOR/2, FACTOR/2)
   input seed should be negative */
float ran1(int idum, int *iv, sycl::nd_item<1> &item)
{
  int j;
  int k;
  int iy = 0;
  int tid = item.get_local_id(0);

  for(j = NTAB; j >=0; j--)      //Load the shuffle
  {
    k = idum / IQ;
    idum = IA * (idum - k * IQ) - IR * k;

    if(idum < 0)
      idum += IM;

    if(j < NTAB)
      iv[NTAB* tid + j] = idum;
  }
  iy = iv[NTAB* tid];

  k = idum / IQ;
  idum = IA * (idum - k * IQ) - IR * k;

  if(idum < 0)
    idum += IM;

  j = iy / NDIV;
  iy = iv[NTAB * tid + j];
  return (AM * iy);  //AM *iy will be between 0.0 and 1.0
}


void kernel_noise_uniform(
  const uchar4* inputImage,
  uchar4* outputImage,
  const int factor,
  int* iv,
  sycl::nd_item<1> &item)
{
  int pos = item.get_global_id(0);

  float4 temp = convert_float4(inputImage[pos]);

  /* compute average value of a pixel from its compoments */
  float avg = (temp.x() + temp.y() + temp.z() + temp.w()) / 4.0f;

  /* Each thread has NTAB private values */

  /* Calculate deviation from the avg value of a pixel */
  float dev = ran1(-avg, iv, item);
  dev = (dev - 0.55f) * (float)factor;

  /* Saturate(clamp) the values */
  outputImage[pos] = convert_uchar4_sat(temp + (float4)(dev));
}
