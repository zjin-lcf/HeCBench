#include "imgTrans.h"
#include "util.h"

int CSCV = 2; // color space conversion: 1 - GPU-native, 2 - GPU-improved

__host__ void Rgb2HSV ( uchar4* inputImg, float4* outputImg, int width, int height )
{
  dim3 ThreadPerBlock(BLOCK_SIZE2,BLOCK_SIZE2);
  dim3 BlockPerGrid(iDivUp(width,BLOCK_SIZE2),iDivUp(height,BLOCK_SIZE2));
  kRgb2HSV<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);
}

__global__ void kRgb2HSV(const uchar4* __restrict__ inputImg, float4* outputImg, int width, int height)
{
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

  uchar4 nPixel=inputImg[offset];

  float _b=(float)nPixel.x/255.0f;
  float _g=(float)nPixel.y/255.0f;
  float _r=(float)nPixel.z/255.0f;

  float max = MAX3(_b, _g, _r);
  float min = MIN3(_b, _g, _r);
  float delta = max - min;

  float4 fPixel;

  if (max == _r)
  {
    fPixel.x = (_g - _b)/delta; 
  }
  else if (max == _g)
  {
    fPixel.x = 2 + (_b - _r)/delta; 
  }
  else if (max == _b)
  {
    fPixel.x = 4 + (_r - _g)/delta; 
  }

  fPixel.x *= 60;

  if(fPixel.x < 0)
  {
    fPixel.x += 360.0f;
  }

  fPixel.x /= 360.0f;

  if (delta == 0)
  {
    fPixel.y = 0;
  }
  else 
  {
    fPixel.y = delta / max;
  }

  fPixel.z = max;

  outputImg[offset] = fPixel;
}

__host__ void Rgb2CIELab( uchar4* inputImg, float4* outputImg, int width, int height )
{
  dim3 ThreadPerBlock(BLOCK_SIZE2,BLOCK_SIZE2);
  dim3 BlockPerGrid(iDivUp(width,BLOCK_SIZE2),iDivUp(height,BLOCK_SIZE2));

  switch (CSCV)
  {
    case 1:
      kRgb2CIELabNative<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);
      break;
    case 2:
      cudaFuncSetCacheConfig(kRgb2CIELabImproved,cudaFuncCachePreferL1);
      kRgb2CIELabImproved<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);
      break;
    default:
      break;
  }
}

__global__ void kRgb2CIELabNative(uchar4* inputImg, float4* outputImg, int width, int height)
{
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

  uchar4 nPixel=inputImg[offset];

  float _b=(float)nPixel.x/255.0;
  float _g=(float)nPixel.y/255.0;
  float _r=(float)nPixel.z/255.0;

  float x=_r*0.412453  +_g*0.357580  +_b*0.180423;
  float y=_r*0.212671  +_g*0.715160  +_b*0.072169;
  float z=_r*0.019334  +_g*0.119193  +_b*0.950227;

  x/=0.950456;
  float y3=exp(log(y)/3.0);
  z/=1.088754;

  float l,a,b;

  x = x>0.008856 ? exp(log(x)/3.0) : (7.787*x+0.13793);
  y = y>0.008856 ? y3 : 7.787*y+0.13793;
  z = z>0.008856 ? z/=exp(log(z)/3.0) : (7.787*z+0.13793);

  l = y>0.008856 ? (116.0*y3-16.0) : 903.3*y;
  a=(x-y)*500.0;
  b=(y-z)*200.0;

  float4 fPixel;
  fPixel.x=l;
  fPixel.y=a;
  fPixel.z=b;

  outputImg[offset]=fPixel;
}

__global__ void kRgb2CIELabImproved(const uchar4* __restrict__ inputImg, float4* outputImg, int width, int height)
{
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

  uchar4 nPixel=inputImg[offset];

  float _b=(float)nPixel.x/255.0f;
  float _g=(float)nPixel.y/255.0f;
  float _r=(float)nPixel.z/255.0f;

  float x=_r*0.412453f  +_g*0.357580f  +_b*0.180423f;
  float y=_r*0.212671f  +_g*0.715160f  +_b*0.072169f;
  float z=_r*0.019334f  +_g*0.119193f  +_b*0.950227f; 

  x/=0.950456f;
  float y3=exp(log(y)/3.0f);
  z/=1.088754f;

  x = x>0.008856f ? exp(log(x)/3.0f) : (7.787f*x+0.13793f);
  y = y>0.008856f ? y3 : 7.787f*y+0.13793f;
  z = z>0.008856f ? z/=exp(log(z)/3.0f) : (7.787f*z+0.13793f);

  outputImg[offset].x = y>0.008856f ? (116.0f*y3-16.0f) : 903.3f*y;
  outputImg[offset].y = (x-y)*500.0f;
  outputImg[offset].z = (y-z)*200.0f;
}


__host__ void Rgb2XYZ( uchar4* inputImg, float4* outputImg, int width, int height )
{
  dim3 ThreadPerBlock(BLOCK_SIZE2,BLOCK_SIZE2);
  dim3 BlockPerGrid(iDivUp(width,BLOCK_SIZE2),iDivUp(height,BLOCK_SIZE2));

  switch (CSCV)
  {
    case 1:
      kRgb2XYZNative<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);  
      break;
    case 2:
      cudaFuncSetCacheConfig(kRgb2CIELabImproved,cudaFuncCachePreferL1);
      kRgb2XYZImproved<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);
      break;
    default:
      break;
  }

}

__global__ void kRgb2XYZNative(uchar4* inputImg, float4* outputImg, int width, int height)
{
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

  uchar4 nPixel=inputImg[offset];

  float _b=(float)nPixel.x/255.0;
  float _g=(float)nPixel.y/255.0;
  float _r=(float)nPixel.z/255.0;

  float x=_r*0.412453  +_g*0.357580  +_b*0.180423;
  float y=_r*0.212671  +_g*0.715160  +_b*0.072169;
  float z=_r*0.019334  +_g*0.119193  +_b*0.950227;

  float4 fPixel;
  fPixel.x=x;
  fPixel.y=y;
  fPixel.z=z;

  outputImg[offset]=fPixel;
}

__global__ void kRgb2XYZImproved(const uchar4* __restrict__ inputImg, float4* outputImg, int width, int height)
{
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

  uchar4 nPixel=inputImg[offset];

  float _b=(float)nPixel.x/255.0f;
  float _g=(float)nPixel.y/255.0f;
  float _r=(float)nPixel.z/255.0f;

  float x=_r*0.412453f  +_g*0.357580f  +_b*0.180423f;
  float y=_r*0.212671f  +_g*0.715160f  +_b*0.072169f;
  float z=_r*0.019334f  +_g*0.119193f  +_b*0.950227f;

  float4 fPixel;
  fPixel.x=x;
  fPixel.y=y;
  fPixel.z=z;

  outputImg[offset]=fPixel;
}


