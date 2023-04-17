#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <chrono>

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

  unsigned int explandedKeySizeBytes = explandedKeySize*sizeof(uchar);

  uchar *expandedKey = (uchar*)malloc(explandedKeySizeBytes);
  uchar *roundKey    = (uchar*)malloc(explandedKeySizeBytes);

  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for(unsigned int i = 0; i < rounds+1; ++i)
  {
    createRoundKey(expandedKey + keySize*i, roundKey + keySize*i);
  }

  // save device result
  uchar* output = (uchar*)malloc(sizeBytes);

  std::cout << "Executing kernel for " << iterations 
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

#pragma omp target data map (to: input[0:sizeBytes], \
                                 roundKey[0:explandedKeySizeBytes], \
                                 sbox[0:256], \
                                 rsbox[0:256]) \
                        map(alloc: output[0:sizeBytes])
{
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    if (decrypt) 
      AESDecrypt(
        (uchar4*)output,
        (uchar4*)input,
        (uchar4*)roundKey,
        rsbox,
        width, height, rounds);
    else
      AESEncrypt(
        (uchar4*)output,
        (uchar4*)input,
        (uchar4*)roundKey,
        sbox,
        width, height, rounds);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

  #pragma omp target update from (output[0:sizeBytes])
}

  // Verify
  uchar *verificationOutput = (uchar *) malloc(sizeBytes);

  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);

  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, sizeBytes) == 0)
    std::cout<<"Pass\n";
  else
    std::cout<<"Fail\n";

  /* release program resources (input memory etc.) */
  if(input) free(input);

  if(key) free(key);

  if(expandedKey) free(expandedKey);

  if(roundKey) free(roundKey);

  if(output) free(output);

  if(verificationOutput) free(verificationOutput);

  return 0;
}
