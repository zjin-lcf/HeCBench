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
#include "common.h"
#include "SDKBitMap.h"
#include "kernels.cpp"

void reference (uchar4 *verificationOutput,
                const uchar4 *inputImageData, 
                const int width,
                const int height,
                const int pixelSize);

static bool compare(const float *refData, const float *data,
    const int size, const float epsilon = 1e-6f)
{
  float error = 0.0f;
  float ref = 0.0f;
  for(int i = 1; i < size; ++i)
  {
    float diff = refData[i] - data[i];
    if (diff != 0) printf("mismatch @%d: %f %f\n", i, refData[i] , data[i]);
    error += diff * diff;
    ref += refData[i] * refData[i];
  }
  float normRef =::sqrtf((float) ref);
  if (::fabs((float) ref) < 1e-7f)
  {
    return false;
  }
  float normError = sqrtf((float) error);
  error = normError / normRef;
  return error < epsilon;
}

int main(int argc, char * argv[])
{
  const char* filePath = argv[1];
  const int iterations = atoi(argv[2]);

  // load input bitmap image
  SDKBitMap inputBitmap;

  inputBitmap.load(filePath);

  // error if image did not load
  if(!inputBitmap.isLoaded())
  {
    printf("Failed to load input image!");
    return SDK_FAILURE;
  }

  // get width and height of input image
  const int height = inputBitmap.getHeight();
  const int width = inputBitmap.getWidth();
  const int pixelSize = sizeof(uchar4);
  const int imageSize = width * height * pixelSize;
  printf("Image height = %d and width = %d\n", height, width);

  // allocate memory for input & output image data
  uchar4 *inputImageData  = (uchar4*)malloc(imageSize);
  if (inputImageData == nullptr)
    printf("Failed to allocate memory! (inputImageData)");

  // allocate memory for output image data
  uchar4 *outputImageData = (uchar4*)malloc(imageSize);
  if (outputImageData == nullptr) 
    printf("Failed to allocate memory! (outputImageData)");

  // initializa the data
  memset(outputImageData, 0, imageSize);

  // get the pointer to pixel data
  uchar4 *pixelData = inputBitmap.getPixels();
  if(pixelData == NULL)
    printf("Failed to read pixel Data!");

  // Copy pixel data into inputImageData
  memcpy(inputImageData, pixelData, imageSize);

  // allocate memory for verification output
  uchar4* verificationOutput = (uchar4*)malloc(imageSize);
  if (verificationOutput == nullptr) 
    printf("verificationOutput heap allocation failed!");

  // initialize the data
  memset(verificationOutput, 0, imageSize);

  {
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    // Create memory object for input Image
    buffer<uchar4, 1> inputImageBuffer (inputImageData, width * height);
    buffer<uchar4, 1> outputImageBuffer (outputImageData, width * height);

    // Enqueue a kernel run call.
    const int blockSizeX = 16;
    const int blockSizeY = 16;
    range<2> gws (height, width);
    range<2> lws (blockSizeY, blockSizeX);

    printf("Executing kernel for %d iterations", iterations);
    printf("-------------------------------------------\n");

    for(int i = 0; i < iterations; i++)
    {
      q.submit([&] (handler &cgh) {
        auto inputImage = inputImageBuffer.get_access<sycl_read>(cgh);
        auto outputImage = outputImageBuffer.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class sobel>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          sobel_filter(inputImage.get_pointer(), 
                       outputImage.get_pointer(), width, height, item);
        });
      });
    }
  }

  // reference implementation
  reference (verificationOutput, inputImageData, width, height, pixelSize);

  float *outputDevice = (float*) malloc (imageSize * sizeof(float));
  if (outputDevice == nullptr)
    printf("Failed to allocate host memory! (outputDevice)");

  float *outputReference = (float*) malloc (imageSize * sizeof(float));

  if (outputReference == nullptr)
    printf("Failed to allocate host memory!" "(outputReference)");

  // copy uchar data to float array
  for(int i = 0; i < (int)(width * height); i++)
  {
    outputDevice[i * 4 + 0] = outputImageData[i].x();
    outputDevice[i * 4 + 1] = outputImageData[i].y();
    outputDevice[i * 4 + 2] = outputImageData[i].z();
    outputDevice[i * 4 + 3] = outputImageData[i].w();

    outputReference[i * 4 + 0] = verificationOutput[i].x();
    outputReference[i * 4 + 1] = verificationOutput[i].y();
    outputReference[i * 4 + 2] = verificationOutput[i].z();
    outputReference[i * 4 + 3] = verificationOutput[i].w();
  }


  // compare the results and see if they match for the given input image
  if(compare(outputReference, outputDevice, imageSize))
    printf("Passed!\n");
  else
    printf("Failed!\n");

  free(outputDevice);
  free(outputReference);
  free(verificationOutput);
  free(inputImageData);
  free(outputImageData);
  return SDK_SUCCESS;
}
