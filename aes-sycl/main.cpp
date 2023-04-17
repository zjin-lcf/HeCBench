#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>

using uchar4 = sycl::uchar4;
using uchar = sycl::uchar;

#include "SDKBitMap.h"
#include "aes.h"
#include "kernels.cpp"
#include "reference.cpp"
#include "utils.cpp"

int main(int argc, char * argv[])
{
  if (argc != 4) {
    printf("Usage: %s <iterations> <0 or 1> <path to bitmap image file>\n", argv[0]);
    printf("0=encrypt, 1=decrypt\n");
    return 1;
  }

  const unsigned int keySizeBits = 128;
  const unsigned int rounds = 10;
  const unsigned int seed = 123;

  const int iterations = atoi(argv[1]);
  const bool decrypt = atoi(argv[2]);
  const char* filePath = argv[3];

  SDKBitMap image;
  image.load(filePath);
  const int width  = image.getWidth();
  const int height = image.getHeight();

  /* check condition for the bitmap to be initialized */
  if (width <= 0 || height <= 0) return 1;

  std::cout << "Image width and height: " 
            << width << " " << height << std::endl;

  uchar4 *pixels = image.getPixels();

  unsigned int sizeBytes = width*height*sizeof(unsigned char);
  unsigned char *input = (unsigned char*)malloc(sizeBytes); 

  /* initialize the input array, do NOTHING but assignment when decrypt*/
  if (decrypt)
    convertGrayToGray(pixels, input, height, width);
  else
    convertColorToGray(pixels, input, height, width);

  unsigned int keySize = keySizeBits/8; // 1 Byte = 8 bits

  unsigned int keySizeBytes = keySize*sizeof(unsigned char);

  unsigned char *key = (unsigned char*)malloc(keySizeBytes);

  fillRandom<unsigned char>(key, keySize, 1, 0, 255, seed); 

  // expand the key
  unsigned int explandedKeySize = (rounds+1)*keySize;
  unsigned char *expandedKey = (unsigned char*)malloc(explandedKeySize*sizeof(unsigned char));
  unsigned char *roundKey    = (unsigned char*)malloc(explandedKeySize*sizeof(unsigned char));

  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for(unsigned int i = 0; i < rounds+1; ++i)
  {
    createRoundKey(expandedKey + keySize*i, roundKey + keySize*i);
  }

  // save device result
  unsigned char* output = (unsigned char*)malloc(sizeBytes);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uchar4 *inputBuffer = (uchar4*) sycl::malloc_device (sizeBytes, q);
  q.memcpy(inputBuffer, input, sizeBytes);

  uchar4 *outputBuffer = (uchar4*) sycl::malloc_device (sizeBytes, q);

  uchar4 *rKeyBuffer = (uchar4*) sycl::malloc_device (explandedKeySize, q);
  q.memcpy(rKeyBuffer, roundKey, explandedKeySize);

  uchar *sBoxBuffer = sycl::malloc_device<uchar>(256, q);
  q.memcpy(sBoxBuffer, sbox, 256);

  uchar *rsBoxBuffer = sycl::malloc_device<uchar>(256, q);
  q.memcpy(rsBoxBuffer, rsbox, 256);

  std::cout << "Executing kernel for " << iterations 
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  sycl::range<2> gws (height, width/4);
  sycl::range<2> lws (4, 1);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    if (decrypt) 
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<uchar4, 1> block0(sycl::range<1>(keySize/4), cgh);
        sycl::local_accessor<uchar4, 1> block1(sycl::range<1>(keySize/4), cgh);
        cgh.parallel_for<class dec>(
          sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          AESDecrypt(outputBuffer,
                     inputBuffer,
                     rKeyBuffer,
                     rsBoxBuffer,
                     block0.get_pointer(),
                     block1.get_pointer(),
                     width, rounds, item);
        });
      });

    else
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<uchar4, 1> block0(sycl::range<1>(keySize/4), cgh);
        sycl::local_accessor<uchar4, 1> block1(sycl::range<1>(keySize/4), cgh);
        cgh.parallel_for<class enc>(
          sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          AESEncrypt(outputBuffer,
                     inputBuffer,
                     rKeyBuffer,
                     sBoxBuffer,
                     block0.get_pointer(),
                     block1.get_pointer(),
                     width, rounds, item);
        });
      });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

  q.memcpy(output, outputBuffer, sizeBytes).wait();

  // Verify
  unsigned char *verificationOutput = (unsigned char *) malloc(sizeBytes);

  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);

  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, sizeBytes) == 0)
    std::cout<<"Pass\n";
  else
    std::cout<<"Fail\n";

  sycl::free(inputBuffer, q);
  sycl::free(outputBuffer, q);
  sycl::free(rKeyBuffer, q);
  sycl::free(sBoxBuffer, q);
  sycl::free(rsBoxBuffer, q);

  /* release program resources (input memory etc.) */
  if(input) free(input);

  if(key) free(key);

  if(expandedKey) free(expandedKey);

  if(roundKey) free(roundKey);

  if(output) free(output);

  if(verificationOutput) free(verificationOutput);

  return 0;
}
