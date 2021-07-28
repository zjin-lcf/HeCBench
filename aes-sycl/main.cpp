#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "SDKBitMap.h"
#include "aes.h"
#include "kernels.cpp"
#include "reference.cpp"
#include "utils.cpp"

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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<unsigned char,1> inputBuffer (input, width * height);
  buffer<unsigned char,1> outputBuffer (width * height);
  buffer<unsigned char,1> rKeyBuffer (roundKey, explandedKeySize);
  buffer<unsigned char,1> sBoxBuffer (sbox, 256);
  buffer<unsigned char,1> rsBoxBuffer (rsbox, 256);

  std::cout << "Executing kernel for " << iterations 
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  range<2> gws (height, width/4);
  range<2> lws (4, 1);
  auto outputBuffer_re = outputBuffer.reinterpret<uchar4>(range<1>(width*height/4));
  auto inputBuffer_re = inputBuffer.reinterpret<uchar4>(range<1>(width*height/4));
  auto rKeyBuffer_re = rKeyBuffer.reinterpret<uchar4>(range<1>(explandedKeySize/4));

  for(int i = 0; i < iterations; i++)
  {
    if (decrypt) 
      q.submit([&] (handler &cgh) {
        auto out = outputBuffer_re.get_access<sycl_discard_write>(cgh);
        auto in = inputBuffer_re.get_access<sycl_read>(cgh);
        auto key = rKeyBuffer_re.get_access<sycl_read>(cgh);
        auto box = rsBoxBuffer.get_access<sycl_read>(cgh);
        accessor<uchar4, 1, sycl_read_write, access::target::local> block0(keySize/4, cgh);
        accessor<uchar4, 1, sycl_read_write, access::target::local> block1(keySize/4, cgh);
        cgh.parallel_for<class dec>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          AESDecrypt(out.get_pointer(),
              in.get_pointer(),
              key.get_pointer(),
              box.get_pointer(),
              block0.get_pointer(),
              block1.get_pointer(),
              width, rounds, item);
        });
      });

    else
      q.submit([&] (handler &cgh) {
        auto out = outputBuffer_re.get_access<sycl_discard_write>(cgh);
        auto in = inputBuffer_re.get_access<sycl_read>(cgh);
        auto key = rKeyBuffer_re.get_access<sycl_read>(cgh);
        auto box = sBoxBuffer.get_access<sycl_read>(cgh);
        accessor<uchar4, 1, sycl_read_write, access::target::local> block0(keySize/4, cgh);
        accessor<uchar4, 1, sycl_read_write, access::target::local> block1(keySize/4, cgh);
        cgh.parallel_for<class enc>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          AESEncrypt(out.get_pointer(),
              in.get_pointer(),
              key.get_pointer(),
              box.get_pointer(),
              block0.get_pointer(),
              block1.get_pointer(),
              width, rounds, item);
        });
      });

    q.submit([&] (handler &cgh) {
      auto acc = outputBuffer.get_access<sycl_read>(cgh);
      cgh.copy(acc, output);
    });
  }
  q.wait();

  // Verify
  unsigned char *verificationOutput = (unsigned char *) malloc(width*height*sizeof(unsigned char));

  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);

  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, height*width*sizeof(unsigned char)) == 0)
    std::cout<<"Passed!\n";
  else
    std::cout<<"Failed\n";

  /* release program resources (input memory etc.) */
  if(input) free(input);

  if(key) free(key);

  if(expandedKey) free(expandedKey);

  if(roundKey) free(roundKey);

  if(output) free(output);

  if(verificationOutput) free(verificationOutput);

  return 0;
}

