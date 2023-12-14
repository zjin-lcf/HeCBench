/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// *********************************************************************
// Demo application for realtime DXT1 compression based on the OpenCL
// DXTC sample
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "dds.h"
#include "permutations.h"
#include "block.h"
#include "shrUtils.h"

#define ERROR_THRESHOLD 0.02f
#define NUM_THREADS     64      // Number of threads per work group.

#include "kernel.cpp"

int main(int argc, char** argv) 
{
  if (argc != 4) {
    printf("Usage: %s <path to image> <path to reference image> <repeat>\n", argv[0]);
    return 1;
  }
  const char* image_path = argv[1];

  const char* reference_image_path = argv[2];

  const int numIterations = atoi(argv[3]);

  unsigned int width, height;
  unsigned int* h_img = NULL;
  const float alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
  const float alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
  const int prods4[4] = {0x090000, 0x000900, 0x040102, 0x010402};
  const int prods3[4] = {0x040000, 0x000400, 0x040101, 0x010401};

  // load image 
  if (!shrLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height)) {
    printf("Error, unable to open source image file <%s>\n", image_path);

    exit(EXIT_FAILURE);
  }

  printf("Image Loaded '%s', %d x %d pixels\n\n", image_path, width, height);

  // Convert linear image to block linear. 
  const unsigned int memSize = width * height;
  const unsigned int memSizeByte = memSize * sizeof(unsigned int);
  unsigned int* block_image = (unsigned int*)malloc(memSizeByte);

  // Convert linear image to block linear. 
  for(unsigned int by = 0; by < height/4; by++) {
    for(unsigned int bx = 0; bx < width/4; bx++) {
      for (int i = 0; i < 16; i++) {
        const int x = i & 3;
        const int y = i / 4;
        block_image[(by * width/4 + bx) * 16 + i] = 
          ((unsigned int *)h_img)[(by * 4 + y) * 4 * (width/4) + bx * 4 + x];
      }
    }
  }

  // Compute permutations.
  unsigned int permutations[1024];
  computePermutations(permutations);

  const unsigned int compressedSize = (width / 4) * (height / 4) * 8;
  unsigned int * h_result = (unsigned int*)malloc(compressedSize);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Tables
  float *d_alphaTable4 = sycl::malloc_device<float>(4, q);
  q.memcpy(d_alphaTable4, alphaTable4, sizeof(float) * 4);

  float *d_alphaTable3 = sycl::malloc_device<float>(4, q);
  q.memcpy(d_alphaTable3, alphaTable3, sizeof(float) * 4);

  int *d_prods4 = sycl::malloc_device<int>(4, q);
  q.memcpy(d_prods4, prods4, sizeof(int) * 4);

  int *d_prods3 = sycl::malloc_device<int>(4, q);
  q.memcpy(d_prods3, prods3, sizeof(int) * 4);

  // Upload permutations.
  unsigned int *d_permutations = sycl::malloc_device<unsigned int>(1024, q);
  q.memcpy(d_permutations, permutations, sizeof(unsigned int) * 1024);

  // Image
  unsigned int *d_image = sycl::malloc_device<unsigned int>(memSize, q);
  q.memcpy(d_image, block_image, sizeof(unsigned int) * memSize);

  // Result 
  sycl::uint2 *d_result = sycl::malloc_device<sycl::uint2>(compressedSize/8, q); 

  // Determine launch configuration and run timed computation numIterations times
  int blocks = ((width + 3) / 4) * ((height + 3) / 4); // rounds up by 1 block in each dim if %4 != 0

  // Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
  int blocksPerLaunch = MIN(blocks, 768 * 24);

  // set work-item dimensions
  size_t szGlobalWorkSize = blocksPerLaunch * NUM_THREADS;

  sycl::range<1> lws (NUM_THREADS);

  printf("\nRunning DXT Compression on %u x %u image...\n", width, height);
  printf("\n%u Workgroups, %u Work Items per Workgroup, %u Work Items in NDRange...\n\n", 
      blocks, NUM_THREADS, blocks * NUM_THREADS);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; ++i) {
    for( int j=0; j<blocks; j+= blocksPerLaunch ) {

      szGlobalWorkSize = MIN( blocksPerLaunch, blocks-j ) * NUM_THREADS;
      sycl::range<1> gws (szGlobalWorkSize);

      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<sycl::float4, 1> s_colors(sycl::range<1>(16), cgh);
        sycl::local_accessor<sycl::float4, 1> s_sums(sycl::range<1>(16), cgh);
        sycl::local_accessor<int, 1> s_int(sycl::range<1>(64), cgh);
        sycl::local_accessor<float, 1> s_float(sycl::range<1>(96), cgh);
        sycl::local_accessor<unsigned int, 1> s_permutations(sycl::range<1>(160), cgh);
        sycl::local_accessor<int, 1> s_xrefs(sycl::range<1>(16), cgh);

        cgh.parallel_for<class dxtc>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
	  const int idx = item.get_local_id(0);
    
          loadColorBlock(item,
                         d_image,
                         s_colors.get_pointer(), 
                         s_sums.get_pointer(),
                         s_xrefs.get_pointer(),
                         s_float.get_pointer(),
                         j);
          
          item.barrier(sycl::access::fence_space::local_space);
          
          sycl::uint4 best = evalAllPermutations(item,
                                                 s_colors.get_pointer(),
                                                 d_permutations,
                                                 s_float.get_pointer(),
                                                 s_sums[0],
                                                 s_permutations.get_pointer(), 
                                                 d_alphaTable4, d_prods4, 
                                                 d_alphaTable3, d_prods3);

          // Use a parallel reduction to find minimum error.
          const int minIdx = findMinError(item, s_float.get_pointer(), s_int.get_pointer());    

          item.barrier(sycl::access::fence_space::local_space);
          
          // Only write the result of the winner thread.
          if (idx == minIdx)
          {
            saveBlockDXT1(item, best.x(), best.y(), best.z(), s_xrefs.get_pointer(), d_result, j);
          }
        });
      });
    }
  }
  
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / numIterations);

  q.memcpy(h_result, (uint*)d_result, compressedSize).wait();

  sycl::free(d_permutations, q);  
  sycl::free(d_image, q);  
  sycl::free(d_result, q);  
  sycl::free(d_alphaTable4, q);
  sycl::free(d_alphaTable3, q);
  sycl::free(d_prods4, q);
  sycl::free(d_prods3, q);
  
  // Write DDS file.
  FILE* fp = NULL;
  char output_filename[1024];
#ifdef WIN32
  strcpy_s(output_filename, 1024, image_path);
  strcpy_s(output_filename + strlen(image_path) - 3, 1024 - strlen(image_path) + 3, "dds");
  fopen_s(&fp, output_filename, "wb");
#else
  strcpy(output_filename, image_path);
  strcpy(output_filename + strlen(image_path) - 3, "dds");
  fp = fopen(output_filename, "wb");
#endif
  assert(fp != NULL);

  DDSHeader header;
  header.fourcc = FOURCC_DDS;
  header.size = 124;
  header.flags  = (DDSD_WIDTH|DDSD_HEIGHT|DDSD_CAPS|DDSD_PIXELFORMAT|DDSD_LINEARSIZE);
  header.height = height;
  header.width = width;
  header.pitch = compressedSize;
  header.depth = 0;
  header.mipmapcount = 0;
  memset(header.reserved, 0, sizeof(header.reserved));
  header.pf.size = 32;
  header.pf.flags = DDPF_FOURCC;
  header.pf.fourcc = FOURCC_DXT1;
  header.pf.bitcount = 0;
  header.pf.rmask = 0;
  header.pf.gmask = 0;
  header.pf.bmask = 0;
  header.pf.amask = 0;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  header.caps.caps2 = 0;
  header.caps.caps3 = 0;
  header.caps.caps4 = 0;
  header.notused = 0;

  fwrite(&header, sizeof(DDSHeader), 1, fp);
  fwrite(h_result, compressedSize, 1, fp);

  fclose(fp);

  // Make sure the generated image matches the reference image (regression check)
  printf("\nComparing against Host/C++ computation...\n");     

  // read in the reference image from file
#ifdef WIN32
  fopen_s(&fp, reference_image_path, "rb");
#else
  fp = fopen(reference_image_path, "rb");
#endif
  assert (fp != NULL);
  fseek(fp, sizeof(DDSHeader), SEEK_SET);
  unsigned int referenceSize = (width / 4) * (height / 4) * 8;
  unsigned int * reference = (unsigned int *)malloc(referenceSize);
  fread(reference, referenceSize, 1, fp);
  fclose(fp);

  // compare the reference image data to the sample/generated image
  float rms = 0;
  for (unsigned int y = 0; y < height; y += 4)
  {
    for (unsigned int x = 0; x < width; x += 4)
    {
      // binary comparison of data
      unsigned int referenceBlockIdx = ((y/4) * (width/4) + (x/4));
      unsigned int resultBlockIdx = ((y/4) * (width/4) + (x/4));
      int cmp = compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx, ((BlockDXT1 *)reference) + referenceBlockIdx);

      // log deviations, if any
      if (cmp != 0.0f) 
      {
        compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx, ((BlockDXT1 *)reference) + referenceBlockIdx);
        //printf("Deviation at (%d, %d):\t%f rms\n", x/4, y/4, float(cmp)/16/3);
      }
      rms += cmp;
    }
  }
  rms /= width * height * 3;
  printf("RMS(reference, result) = %f\n\n", rms);

  // Free host memory
  free(block_image);
  free(h_result);
  free(h_img);
  free(reference);

  // finish
  if (rms <= ERROR_THRESHOLD) 
    printf("PASS\n"); 
  else
    printf("FAIL\n");

  return 0;
}
