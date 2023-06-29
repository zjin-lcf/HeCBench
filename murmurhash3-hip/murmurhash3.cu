//-------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
//-------------------------------------------------------------------------

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

#define FORCE_INLINE inline __attribute__((always_inline))

__host__ __device__
inline uint64_t rotl64 ( uint64_t x, int8_t r )
{
  return (x << r) | (x >> (64 - r));
}

#define BIG_CONSTANT(x) (x##LU)

// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here
__host__ __device__
FORCE_INLINE uint64_t getblock64 ( const uint8_t * p, uint32_t i )
{
  uint64_t s = 0;
  for (uint32_t n = 0; n < 8; n++) {
    s |= ((uint64_t)p[8*i+n] << (n*8));
  }
  return s;
}

// Finalization mix - force all bits of a hash block to avalanche
__host__ __device__
FORCE_INLINE uint64_t fmix64 ( uint64_t k )
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__host__ __device__ 
void MurmurHash3_x64_128 (const void * key, const uint32_t len,
                          const uint32_t seed, void * out)
{
  const uint8_t * data = (const uint8_t*)key;
  const uint32_t nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  for(uint32_t i = 0; i < nblocks; i++)
  {
    uint64_t k1 = getblock64(data,i*2+0);
    uint64_t k2 = getblock64(data,i*2+1);

    k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;

    h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

    k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

    h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
  }

  const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch(len & 15)
  {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48;
    case 14: k2 ^= ((uint64_t)tail[13]) << 40;
    case 13: k2 ^= ((uint64_t)tail[12]) << 32;
    case 12: k2 ^= ((uint64_t)tail[11]) << 24;
    case 11: k2 ^= ((uint64_t)tail[10]) << 16;
    case 10: k2 ^= ((uint64_t)tail[ 9]) << 8;
    case  9: k2 ^= ((uint64_t)tail[ 8]) << 0;
       k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

    case  8: k1 ^= ((uint64_t)tail[ 7]) << 56;
    case  7: k1 ^= ((uint64_t)tail[ 6]) << 48;
    case  6: k1 ^= ((uint64_t)tail[ 5]) << 40;
    case  5: k1 ^= ((uint64_t)tail[ 4]) << 32;
    case  4: k1 ^= ((uint64_t)tail[ 3]) << 24;
    case  3: k1 ^= ((uint64_t)tail[ 2]) << 16;
    case  2: k1 ^= ((uint64_t)tail[ 1]) << 8;
    case  1: k1 ^= ((uint64_t)tail[ 0]) << 0;
       k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
  };

  h1 ^= len; h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  ((uint64_t*)out)[0] = h1;
  ((uint64_t*)out)[1] = h2;
}

__global__
void MurmurHash3_x64_128_kernel (
   const uint8_t *__restrict__ d_keys,
   const uint32_t *__restrict__ d_length,
   const uint32_t *__restrict__ length, 
         uint64_t *__restrict__ d_out,
   const uint32_t numKeys )
{
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numKeys) 
    MurmurHash3_x64_128 (d_keys+d_length[i], length[i], i, d_out+i*2);
}

int main(int argc, char** argv) 
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  } 
  uint32_t numKeys = atoi(argv[1]);
  uint32_t repeat = atoi(argv[2]);

  srand(3);
  uint32_t i;
  // length of each key
  uint32_t* length = (uint32_t*) malloc (sizeof(uint32_t) * numKeys);
  // pointer to each key
  uint8_t** keys = (uint8_t**) malloc (sizeof(uint8_t*) * numKeys);
  // hashing output
  uint64_t** out = (uint64_t**) malloc (sizeof(uint64_t*) * numKeys);

  for (i = 0; i < numKeys; i++) {
    length[i] = rand() % 10000;
    keys[i] = (uint8_t*) malloc (length[i]);
    out[i] = (uint64_t*) malloc (2*sizeof(uint64_t));
    for (uint32_t c = 0; c < length[i]; c++) {
      keys[i][c] = c % 256;
    }
    MurmurHash3_x64_128 (keys[i], length[i], i, out[i]);
#ifdef DEBUG
    printf("%lu %lu\n", out[i][0], out[i][1]);
#endif
  }

  // create the 1D data arrays for device offloading  
  uint64_t* d_out = (uint64_t*) malloc (sizeof(uint64_t) * 2 * numKeys);
  uint32_t* d_length = (uint32_t*) malloc (sizeof(uint32_t) * (numKeys+1));

  // initialize the length array
  uint32_t total_length = 0;
  d_length[0] = 0;
  for (uint32_t i = 0; i < numKeys; i++) {
    total_length += length[i];
    d_length[i+1] = total_length;
  }

  // initialize the key array 
  uint8_t* d_keys = (uint8_t*) malloc (sizeof(uint8_t) * total_length);

  for (uint32_t i = 0; i < numKeys; i++) {
    memcpy(d_keys+d_length[i], keys[i], length[i]);
  }

  // sanity check
  for (uint32_t i = 0; i < numKeys; i++) {
    assert (0 == memcmp(d_keys+d_length[i], keys[i], length[i]));
  }

  uint8_t* dev_keys;
  hipMalloc((void**)&dev_keys, sizeof(uint8_t)*total_length);
  hipMemcpy(dev_keys, d_keys, sizeof(uint8_t)*total_length, hipMemcpyHostToDevice);

  uint32_t* dev_length;
  hipMalloc((void**)&dev_length, sizeof(uint32_t)*(numKeys+1));
  hipMemcpy(dev_length, d_length, sizeof(uint32_t)*(numKeys+1), hipMemcpyHostToDevice);

  uint32_t* key_length;
  hipMalloc((void**)&key_length, sizeof(uint32_t)*(numKeys+1));
  hipMemcpy(key_length, length, sizeof(uint32_t)*(numKeys), hipMemcpyHostToDevice);

  uint64_t* dev_out;
  hipMalloc((void**)&dev_out, sizeof(uint64_t)*(numKeys*2));

  dim3 gridDim ((numKeys+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (uint32_t n = 0; n < repeat; n++)  
    hipLaunchKernelGGL(MurmurHash3_x64_128_kernel, gridDim, blockDim, 0, 0, 
      dev_keys, dev_length, key_length, dev_out, numKeys);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(d_out, dev_out, sizeof(uint64_t)*(numKeys*2), hipMemcpyDeviceToHost);

  // verify
  bool error = false;
  for (uint32_t i = 0; i < numKeys; i++) {
    if (d_out[2*i] != out[i][0] ||  d_out[2*i+1] != out[i][1]) {
      error = true;
      break;
    }
  }
  if (error) printf("FAIL\n");
  else printf("SUCCESS\n");

  for (uint32_t i = 0; i < numKeys; i++) {
    free(out[i]);
    free(keys[i]);
  }
  free(keys);
  free(out);
  free(length);
  free(d_keys);
  free(d_out);
  free(d_length);
  hipFree(dev_keys);
  hipFree(dev_out);
  hipFree(dev_length);
  hipFree(key_length);
  return 0;
}
