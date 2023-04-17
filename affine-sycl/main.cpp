/**********
  Copyright (c) 2017, Xilinx, Inc.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********/
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include "reference.h"

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    printf("Usage: %s <input image> <output image> <iterations>\n", argv[0]) ;
    return -1 ;
  }

  unsigned short input_image[Y_SIZE*X_SIZE] __attribute__((aligned(1024)));
  unsigned short output_image[Y_SIZE*X_SIZE] __attribute__((aligned(1024)));
  unsigned short output_image_ref[Y_SIZE*X_SIZE] __attribute__((aligned(1024)));

  // Read the bit map file into memory and allocate memory for the final image
  std::cout << "Reading input image...\n";

  // Load the input image
  const char *inputImageFilename = argv[1];
  FILE *input_file = fopen(inputImageFilename, "rb");
  if (!input_file)
  {
    printf("Error: Unable to open input image file %s!\n", inputImageFilename);
    return 1;
  }

  printf("\n");
  printf("   Reading RAW Image\n");
  size_t items_read = fread(input_image, sizeof(input_image), 1, input_file);
  printf("   Bytes read = %d\n\n", (int)(items_read * sizeof(input_image)));
  fclose(input_file);

  const int iterations = atoi(argv[3]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t image_size_bytes = sizeof(unsigned short) * X_SIZE * Y_SIZE;
  
  unsigned short *d_input_image = sycl::malloc_device<unsigned short>(X_SIZE * Y_SIZE, q);
  q.memcpy(d_input_image, input_image, image_size_bytes); 

  unsigned short *d_output_image = sycl::malloc_device<unsigned short>(X_SIZE*Y_SIZE, q);

  sycl::range<2> globalSize(Y_SIZE,X_SIZE);
  sycl::range<2> localSize(16,16);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class affine_transform> (
        sycl::nd_range<2>(globalSize, localSize), [=](sycl::nd_item<2> item) {
        int y = item.get_global_id(0); 
        int x = item.get_global_id(1); 

        const float lx_rot   = 30.0f;
        const float ly_rot   = 0.0f; 
        const float lx_expan = 0.5f;
        const float ly_expan = 0.5f; 
        int   lx_move  = 0;
        int   ly_move  = 0;
        float affine[2][2];   // coefficients
        float i_affine[2][2];
        float beta[2];
        float i_beta[2];
        float det;
        float x_new, y_new;
        float x_frac, y_frac;
        float gray_new;
        int   m, n;
        unsigned short output_buffer;

        // forward affine transformation 
        affine[0][0] = lx_expan * sycl::cos(lx_rot*PI/180.0f);
        affine[0][1] = ly_expan * sycl::sin(ly_rot*PI/180.0f);
        affine[1][0] = lx_expan * sycl::sin(lx_rot*PI/180.0f);
        affine[1][1] = ly_expan * sycl::cos(ly_rot*PI/180.0f);
        beta[0]      = lx_move;
        beta[1]      = ly_move;

        // determination of inverse affine transformation
        det = (affine[0][0] * affine[1][1]) - (affine[0][1] * affine[1][0]);
        if (det == 0.0f)
        {
          i_affine[0][0] = 1.0f;
          i_affine[0][1] = 0.0f;
          i_affine[1][0] = 0.0f;
          i_affine[1][1] = 1.0f;
          i_beta[0]      = -beta[0];
          i_beta[1]      = -beta[1];
        } 
        else 
        {
          i_affine[0][0] =  affine[1][1]/det;
          i_affine[0][1] = -affine[0][1]/det;
          i_affine[1][0] = -affine[1][0]/det;
          i_affine[1][1] =  affine[0][0]/det;
          i_beta[0]      = -i_affine[0][0]*beta[0]-i_affine[0][1]*beta[1];
          i_beta[1]      = -i_affine[1][0]*beta[0]-i_affine[1][1]*beta[1];
        }

        // Output image generation by inverse affine transformation and bilinear transformation

        x_new  = i_beta[0] + i_affine[0][0]*(x-X_SIZE/2.0f) + i_affine[0][1]*(y-Y_SIZE/2.0f) + X_SIZE/2.0f;
        y_new  = i_beta[1] + i_affine[1][0]*(x-X_SIZE/2.0f) + i_affine[1][1]*(y-Y_SIZE/2.0f) + Y_SIZE/2.0f;

        m      = (int)sycl::floor(x_new);
        n      = (int)sycl::floor(y_new);

        x_frac = x_new - m;
        y_frac = y_new - n;

        if ((m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n+1 < Y_SIZE))
        {
          gray_new = (1.0f - y_frac) * ((1.0f - x_frac) * d_input_image[(n * X_SIZE) + m] +
                     x_frac * d_input_image[(n * X_SIZE) + m + 1]) +
                     y_frac  * ((1.0f - x_frac) * (d_input_image[((n + 1) * X_SIZE) + m]) +
                     x_frac * (d_input_image[((n + 1) * X_SIZE) + m + 1]));

          output_buffer = (unsigned short)gray_new;
        } 
        else if (((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) || ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE))) 
        {
          output_buffer = d_input_image[(n * X_SIZE) + m];
        } 
        else 
        {
          output_buffer = WHITE;
        }

        d_output_image[(y * X_SIZE)+x] = output_buffer;
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "   Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

  q.memcpy(output_image, d_output_image, image_size_bytes).wait();

  sycl::free(d_input_image, q);
  sycl::free(d_output_image, q);

  // verify
  affine_reference(input_image, output_image_ref);
  int max_error = 0;
  for (int y = 0; y < Y_SIZE; y++) {
    for (int x = 0; x < X_SIZE; x++) {
      max_error = std::max(max_error, std::abs(output_image[y*X_SIZE+x] - output_image_ref[y*X_SIZE+x]));
    }
  }
  printf("   Max output error is %d\n\n", max_error);

  printf("   Writing RAW Image\n");
  const char *outputImageFilename = argv[2];
  FILE *output_file = fopen(outputImageFilename, "wb");
  if (!output_file)
  {
    printf("Error: Unable to write  image file %s!\n", outputImageFilename);
    return 1;
  }
  size_t items_written = fwrite(output_image, sizeof(output_image), 1, output_file);
  printf("   Bytes written = %d\n\n", (int)(items_written * sizeof(output_image)));
  fclose(output_file);

  return 0 ;
}
