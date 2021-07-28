#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "SDKBitMap.h"
#include "aes.h"
#include "kernels.cu"
#include "reference.cu"
#include "utils.cu"

int main(int argc, char * argv[])
{
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

  unsigned int sizeBytes = width*height*sizeof(uchar);
  uchar *input = (uchar*)malloc(sizeBytes); 

  /* initialize the input array, do NOTHING but assignment when decrypt*/
  if (decrypt)
    convertGrayToGray(pixels, input, height, width);
  else
    convertColorToGray(pixels, input, height, width);

  unsigned int keySize = keySizeBits/8; // 1 Byte = 8 bits

  unsigned int keySizeBytes = keySize*sizeof(uchar);

  uchar *key = (uchar*)malloc(keySizeBytes);

  fillRandom<uchar>(key, keySize, 1, 0, 255, seed); 

  // expand the key
  unsigned int explandedKeySize = (rounds+1)*keySize;
  uchar *expandedKey = (uchar*)malloc(explandedKeySize*sizeof(uchar));
  uchar *roundKey    = (uchar*)malloc(explandedKeySize*sizeof(uchar));

  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for(unsigned int i = 0; i < rounds+1; ++i)
  {
    createRoundKey(expandedKey + keySize*i, roundKey + keySize*i);
  }

  // save device result
  uchar* output = (uchar*)malloc(sizeBytes);

  uchar *inputBuffer;
  cudaMalloc((void**)&inputBuffer, width * height);
  cudaMemcpy(inputBuffer, input, width * height, cudaMemcpyHostToDevice);

  uchar *outputBuffer;
  cudaMalloc((void**)&outputBuffer, width * height);

  uchar *rKeyBuffer;
  cudaMalloc((void**)&rKeyBuffer, explandedKeySize);
  cudaMemcpy(rKeyBuffer, roundKey, explandedKeySize, cudaMemcpyHostToDevice);

  uchar *sBoxBuffer;
  cudaMalloc((void**)&sBoxBuffer, 256);
  cudaMemcpy(sBoxBuffer, sbox, 256, cudaMemcpyHostToDevice);

  uchar *rsBoxBuffer;
  cudaMalloc((void**)&rsBoxBuffer, 256);
  cudaMemcpy(rsBoxBuffer, rsbox, 256, cudaMemcpyHostToDevice);

  std::cout << "Executing kernel for " << iterations 
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  dim3 grid (width/4, height/4);
  dim3 block (1, 4);

  for(int i = 0; i < iterations; i++)
  {
    if (decrypt) 
      AESDecrypt<<< grid, block >>>(
        (uchar4*)outputBuffer,
        (uchar4*)inputBuffer,
        (uchar4*)rKeyBuffer,
        rsBoxBuffer,
        width, rounds);
    else
      AESEncrypt<<< grid, block >>>(
        (uchar4*)outputBuffer,
        (uchar4*)inputBuffer,
        (uchar4*)rKeyBuffer,
        sBoxBuffer,
        width, rounds);

    cudaMemcpy(output, outputBuffer, width * height, cudaMemcpyDeviceToHost);
  }

  // Verify
  uchar *verificationOutput = (uchar *) malloc(width*height*sizeof(uchar));

  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);

  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, height*width*sizeof(uchar)) == 0)
    std::cout<<"Passed!\n";
  else
    std::cout<<"Failed\n";

  /* release program resources (input memory etc.) */
  cudaFree(inputBuffer);
  cudaFree(outputBuffer);
  cudaFree(rKeyBuffer);
  cudaFree(sBoxBuffer);
  cudaFree(rsBoxBuffer);

  if(input) free(input);

  if(key) free(key);

  if(expandedKey) free(expandedKey);

  if(roundKey) free(roundKey);

  if(output) free(output);

  if(verificationOutput) free(verificationOutput);

  return 0;
}

