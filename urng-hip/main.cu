/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/


#include <cmath>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>
#include "urng.h"
#include "kernel.cu"

int main(int argc, char** argv) 
{
  const char* filePath = argv[1];
  const int blockSizeX = atoi(argv[2]);
  const int blockSizeY = atoi(argv[3]);
  const int iterations = atoi(argv[4]);

  // load input bitmap image
  SDKBitMap inputBitmap;   
  inputBitmap.load(filePath);
  if(!inputBitmap.isLoaded())
  {
    std::cout << "Failed to load input image!";
    return -1;
  }

  // get width and height of input image
  int height = inputBitmap.getHeight();
  int width = inputBitmap.getWidth();
  uint pixelSize = sizeof(uchar4);

  std::cout << "Image " << filePath;
  std::cout << " height: " << height;
  std::cout << " width: " << width << std::endl;

  // allocate memory for input & output image data
  uchar4* inputImageData  = (uchar4*)malloc(width * height * sizeof(uchar4));

  // allocate memory for output image data
  uchar4* outputImageData = (uchar4*)malloc(width * height * sizeof(uchar4));

  // initializa the Image data to NULL
  memset(outputImageData, 0, width * height * pixelSize);

  // get the pointer to pixel data
  uchar4 *pixelData = inputBitmap.getPixels();
  if(pixelData == NULL)
  {
    std::cout << "Failed to read pixel Data!";
    free(inputImageData);
    free(outputImageData);
    return -1;
  }

  // Copy pixel data into inputImageData
  memcpy(inputImageData, pixelData, width * height * pixelSize);

  // allocate memory for verification output
  uchar4 *verificationOutput = (uchar4*)malloc(width * height * pixelSize);

  // initialize the data to NULL
  memset(verificationOutput, 0, width * height * pixelSize);

  const int factor = FACTOR;

  uchar4 *inputImageBuffer;
  hipMalloc((void**)&inputImageBuffer, width * height * sizeof(uchar4)); 

  uchar4 *outputImageBuffer;
  hipMalloc((void**)&outputImageBuffer, width * height * sizeof(uchar4)); 

  dim3 grid (height * width / (blockSizeY * blockSizeX));
  dim3 block (blockSizeY * blockSizeX);  // maximum work-group size is 256

  std::cout << "Executing kernel for " << iterations << " iterations" <<std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  hipMemcpy(inputImageBuffer, inputImageData, width * height * sizeof(uchar4), hipMemcpyHostToDevice);
  for(int i = 0; i < iterations; i++)
  {
    hipLaunchKernelGGL(noise_uniform, dim3(grid), dim3(block), 0, 0, inputImageBuffer, outputImageBuffer, factor);
  }
  hipMemcpy(outputImageData, outputImageBuffer, width * height * sizeof(uchar4), hipMemcpyDeviceToHost);
  hipFree(inputImageBuffer);
  hipFree(outputImageBuffer);

  // verify
  float mean = 0;
  for(int i = 0; i < (int)(width * height); i++)
  {
    mean += outputImageData[i].x - inputImageData[i].x;
    mean += outputImageData[i].y - inputImageData[i].y;
    mean += outputImageData[i].z - inputImageData[i].z;
    mean += outputImageData[i].w - inputImageData[i].w;
  }
  mean /= (4 * width * height * factor);
  std::cout << "The averaged mean: " << mean << std::endl;

  if(fabs(mean) < 1.0)
  {
    std::cout << "Passed! \n" << std::endl;
  }
  else
  {
    std::cout << "Failed! \n" << std::endl;
  }

#ifdef DUMP
  // copy output image data back to original pixel data
  memcpy(pixelData, outputImageData, width * height * pixelSize);

  // write the output bmp file
  if(!inputBitmap.write(OUTPUT_IMAGE))
    std::cout << "Failed to write output image!";
  else 
    std::cout << "Write output image!";
#endif

  // release program resources (input memory etc.)
  free(inputImageData);
  free(outputImageData);
  free(verificationOutput);
  return 0;
}


