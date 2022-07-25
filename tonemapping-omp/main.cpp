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

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <math.h>

#pragma omp declare target
inline float luminance(float r, float g, float b)
{
  return ( 0.2126f * r ) + ( 0.7152f * g ) + ( 0.0722f * b );
}
#pragma omp end declare target

double runKernels(
    const float *input,
    float *output,
    const float averageLuminance, 
    const float gamma, 
    const float c, 
    const float delta,
    const uint width,
    const uint numChannels,
    const uint height)
{
  #pragma omp target update to (input[0:width*numChannels*height])

  auto start = std::chrono::steady_clock::now();

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (uint y = 0; y < height; y++) {
    for (uint x = 0; x < width; x++) {
      float r, g, b;
      float cLPattanaik;
      float yLPattanaik;

      float r1 = input[width * numChannels * y + (x * numChannels + 0)];
      float g1 = input[width * numChannels * y + (x * numChannels + 1)];
      float b1 = input[width * numChannels * y + (x * numChannels + 2)];

      float yLuminance = luminance(r1, g1, b1);
      float gcPattanaik = c * averageLuminance;

      if (x != 0 && y != 0 && x != width-1 && y != height-1)
      {
        //Calculating mean
        float leftUp = 0.0f;
        float up = 0.0f;
        float rightUp = 0.0f;
        float left = 0.0f;
        float right = 0.0f;
        float leftDown = 0.0f;
        float down = 0.0f;
        float rightDown = 0.0f;

        r = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 0];
        g = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 1];
        b = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 2];

        leftUp = luminance( r, g, b );

        r = input[width * numChannels * (y - 1) + ((x) * numChannels) + 0];
        g = input[width * numChannels * (y - 1) + ((x) * numChannels) + 1];
        b = input[width * numChannels * (y - 1) + ((x) * numChannels) + 2];

        up = luminance( r, g, b );

        r = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 0];
        g = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 1];
        b = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 2];

        rightUp = luminance( r, g, b );

        r = input[width * numChannels * (y) + ((x - 1) * numChannels) + 0];
        g = input[width * numChannels * (y) + ((x - 1) * numChannels) + 1];
        b = input[width * numChannels * (y) + ((x - 1) * numChannels) + 2];

        left = luminance( r, g, b );  

        r = input[width * numChannels * (y) + ((x + 1) * numChannels) + 0];
        g = input[width * numChannels * (y) + ((x + 1) * numChannels) + 1];
        b = input[width * numChannels * (y) + ((x + 1) * numChannels) + 2];

        right = luminance( r, g, b );  

        r = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 0];
        g = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 1];
        b = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 2];

        leftDown = luminance( r, g, b );

        r = input[width * numChannels * (y + 1) + ((x) * numChannels) + 0];
        g = input[width * numChannels * (y + 1) + ((x) * numChannels) + 1];
        b = input[width * numChannels * (y + 1) + ((x) * numChannels) + 2];

        down = luminance( r, g, b );

        r = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 0];
        g = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 1];
        b = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 2];

        rightDown = luminance( r, g, b );

        //Calculate median    
        yLPattanaik = (leftUp + up + rightUp + left + right + leftDown + down + rightDown) / 8;    
      }
      else
      {
        yLPattanaik = yLuminance;
      }

      cLPattanaik =  yLPattanaik * logf(delta + yLPattanaik / yLuminance) + gcPattanaik;


      float yDPattanaik = yLuminance / (yLuminance + cLPattanaik);

      r = powf((r1 / yLuminance), gamma) * yDPattanaik;
      g = powf((g1 / yLuminance), gamma) * yDPattanaik;  
      b = powf((b1 / yLuminance), gamma) * yDPattanaik;

      output[width * numChannels * y + (x * numChannels + 0)] = r;
      output[width * numChannels * y + (x * numChannels + 1)] = g;
      output[width * numChannels * y + (x * numChannels + 2)] = b;
      output[width * numChannels * y + (x * numChannels + 3)] = 
        input[width * numChannels * y + (x * numChannels + 3)];
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  #pragma omp target update from (output[0:width*numChannels*height])

  return time;
}


int main(int argc, char *argv[])
{
  const char* inputImageName = argv[1]; //"input.hdr";
  const int iterations = atoi(argv[2]);
  const float cPattanaik = 0.25f;
  const float gammaPattanaik = 0.4f;
  const float deltaPattanaik = 0.000002f;
  const uint numChannels = 4;
  uint width;
  uint height;
  float averageLuminance = 0.0f;

  // Read a simple image 
  std::ifstream inputFile;
  std::cout << "Input file name " << inputImageName << std::endl;
  inputFile.open(inputImageName, std::ifstream::binary);

  if (!inputFile.is_open())
  {
    std::cout << "not able to open the file  " << inputImageName << std::endl;
    return 1;
  }

  inputFile >> width;
  inputFile >> height;

  float *input = (float*) aligned_alloc(1024, height * width * sizeof(float) * numChannels);
  float *output = (float*) aligned_alloc(1024, height * width * sizeof(float) * numChannels);
  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      inputFile >> input[(y * width * numChannels) + (x * numChannels + 0)];
      inputFile >> input[(y * width * numChannels) + (x * numChannels + 1)];
      inputFile >> input[(y * width * numChannels) + (x * numChannels + 2)];
      inputFile >> input[(y * width * numChannels) + (x * numChannels + 3)];
    }
  }
  std::cout << "Width of the image " << width << std::endl;
  std::cout << "Height of the image " << height << std::endl;
  inputFile.close();

  //Calculating average luminance value
  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      float r = input[(y * width * numChannels) + (x * numChannels + 0)];
      float g = input[(y * width * numChannels) + (x * numChannels + 1)];
      float b = input[(y * width * numChannels) + (x * numChannels + 2)];

      float luminance = (0.2126f * r ) + ( 0.7152f * g ) + ( 0.0722f * b );
      averageLuminance += luminance;
    }
  }

  averageLuminance = averageLuminance / (width * height);
  std::cout << "Average luminance value in the image " 
    << averageLuminance << std::endl;


#pragma omp target data map(alloc: input[0:width*numChannels*height], \
	                           output[0:width*numChannels*height])
{
  // Warm up
  for(int i = 0; i < 2 && iterations != 1; i++)
  {
    runKernels(
      input,
      output,
      averageLuminance, 
      gammaPattanaik, 
      cPattanaik, 
      deltaPattanaik, 
      width, 
      numChannels, 
      height);
  }
  std::cout << "Executing kernel for " << iterations << " iterations" <<std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  double time = 0.0;

  for(int i = 0; i < iterations; i++)
  {
    time += runKernels(
      input,
      output,
      averageLuminance, 
      gammaPattanaik, 
      cPattanaik, 
      deltaPattanaik, 
      width, 
      numChannels, 
      height);
  }

  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);
}

  // VerifyResults
  float *referenceOutput = (float*) malloc (sizeof(float) * height * width * numChannels);

  float gcPattanaik = cPattanaik * averageLuminance;

  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      float yLPattanaik = 0.0f;
      float cLPattanaik = 0.0f;

      float r, g, b;
      //Finding R, G, B values of a pixel
      float r1 = input[y * width * numChannels + (x * numChannels + 0)];
      float g1 = input[y * width * numChannels + (x * numChannels + 1)];
      float b1 = input[y * width * numChannels + (x * numChannels + 2)];
      //Calculating the luminance value
      float yLuminance = (0.2126f * r1) + (0.7152f * g1) + (0.0722f * b1);

      if (x != 0 && y != 0 && x != width - 1 && y != height - 1)
      {
        //Calculating mean
        float leftUp = 0.0f;
        float up = 0.0f;
        float rightUp = 0.0f;
        float left = 0.0f;
        float right = 0.0f;
        float leftDown = 0.0f;
        float down = 0.0f;
        float rightDown = 0.0f;

        r = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 2 ];

        leftUp = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y - 1) + ((x) * numChannels) + 0 ];
        g = input[width * numChannels * (y - 1) + ((x) * numChannels) + 1 ];
        b = input[width * numChannels * (y - 1) + ((x) * numChannels) + 2 ];

        up = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 2 ];

        rightUp = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y) + ((x - 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y) + ((x - 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y) + ((x - 1) * numChannels) + 2 ];

        left = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y) + ((x + 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y) + ((x + 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y) + ((x + 1) * numChannels) + 2 ];

        right = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 2 ];

        leftDown = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y + 1) + ((x) * numChannels) + 0 ];
        g = input[width * numChannels * (y + 1) + ((x) * numChannels) + 1 ];
        b = input[width * numChannels * (y + 1) + ((x) * numChannels) + 2 ];

        down = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 2 ];

        rightDown = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        //Calculate median

        yLPattanaik = (leftUp + up + rightUp + left + right + leftDown + down +
            rightDown) / 8;
      }
      else
      {
        yLPattanaik = yLuminance;
      }

      cLPattanaik =  yLPattanaik * log(deltaPattanaik + yLPattanaik / yLuminance) +
        gcPattanaik;

      float yDPattanaik = yLuminance / (yLuminance + cLPattanaik);

      r = pow((r1 / yLuminance), gammaPattanaik) * yDPattanaik;
      g = pow((g1 / yLuminance), gammaPattanaik) * yDPattanaik;
      b = pow((b1 / yLuminance), gammaPattanaik) * yDPattanaik;

      referenceOutput[width * numChannels * y + (x * numChannels + 0)] = r;
      referenceOutput[width * numChannels * y + (x * numChannels + 1)] = g;
      referenceOutput[width * numChannels * y + (x * numChannels + 2)] = b;
      referenceOutput[width * numChannels * y + (x * numChannels + 3)] =
        input[width * numChannels * y + (x * numChannels + 3)];
    }
  }

  float error = 0.0f;

  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      error += referenceOutput[width * numChannels * y + (x * numChannels + 0)] -
        output[width * numChannels * y + (x * numChannels + 0)];
      error += referenceOutput[width * numChannels * y + (x * numChannels + 1)] -
        output[width * numChannels * y + (x * numChannels + 1)];
      error += referenceOutput[width * numChannels * y + (x * numChannels + 2)] -
        output[width * numChannels * y + (x * numChannels + 2)];
      error += referenceOutput[width * numChannels * y + (x * numChannels + 3)] -
        output[width * numChannels * y + (x * numChannels + 3)];
    }
  }

  error = error / (height * width);

  if(error > 0.000001f)
  {
    std::cout << "FAIL with normalized error: " << error << std::endl;
    return 1;
  }
  else
  {
    std::cout << "PASS" << std::endl;
  }

  free(input);
  free(output);
  free(referenceOutput);
  return 0;
}
