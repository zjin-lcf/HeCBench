#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "common.h"
#include "chacha20.h"

void hex_to_raw(nd_item<1> &item, const char* src, const int n /*src size*/,
                uint8_t* dst, const uint8_t* char_to_uint)
{
  for (int i = item.get_local_id(0); i < n/2; i = i + item.get_local_range(0)) {
    uint8_t hi = char_to_uint[src[i*2 + 0]];
    uint8_t lo = char_to_uint[src[i*2 + 1]];
    dst[i] = (hi << 4) | lo;
  }
}

void test_keystreams (
    nd_item<1> &item,
    const char *__restrict text_key,
    const char *__restrict text_nonce,
    const char *__restrict text_keystream,
    const uint8_t *__restrict char_to_uint,
    uint8_t *__restrict raw_key,
    uint8_t *__restrict raw_nonce,
    uint8_t *__restrict raw_keystream,
    uint8_t *__restrict result,
    const int text_key_size,
    const int text_nonce_size,
    const int text_keystream_size)

{
  hex_to_raw(item, text_key, text_key_size, raw_key, char_to_uint);
  hex_to_raw(item, text_nonce, text_nonce_size, raw_nonce, char_to_uint);
  hex_to_raw(item, text_keystream, text_keystream_size, raw_keystream, char_to_uint);
   
  if (item.get_local_id(0) == 0) {
    Chacha20 chacha(raw_key, raw_nonce);
    chacha.crypt(result, text_keystream_size / 2);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // Initialize lookup table
  uint8_t char_to_uint[256];
  for (int i = 0; i < 10; i++) char_to_uint[i + '0'] = i;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'a'] = i + 10;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'A'] = i + 10;

  // Sample
  const char *h_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  const char *h_nonce = "0001020304050607";
  const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

  const int key_len = strlen(h_key);
  const int nonce_len = strlen(h_nonce);
  const int keystream_len = strlen(h_keystream);
  const int result_len = keystream_len / 2;

  uint8_t *result = (uint8_t*) malloc (result_len);
  uint8_t *raw_keystream = (uint8_t*) malloc (result_len);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<uint8_t, 1> d_char_to_uint (char_to_uint, 256);

  buffer<char, 1> d_key (h_key, key_len);
  buffer<uint8_t, 1> d_raw_key (key_len/2); 

  buffer<char, 1> d_nonce (h_nonce, nonce_len); 
  buffer<uint8_t, 1> d_raw_nonce (nonce_len/2); 

  buffer<char, 1> d_keystream (h_keystream, keystream_len);

  buffer<uint8_t, 1> d_raw_keystream (result_len); 
  buffer<uint8_t, 1> d_result (result_len); 

  range<1> gws (256);
  range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto acc = d_result.get_access<sycl_write>(cgh);
      cgh.fill(acc, (uint8_t)0);
    });

    q.submit([&] (handler &cgh) {
      auto key = d_key.get_access<sycl_read>(cgh);
      auto nonce = d_nonce.get_access<sycl_read>(cgh);
      auto keystream = d_keystream.get_access<sycl_read>(cgh);
      auto char2uint = d_char_to_uint.get_access<sycl_read>(cgh);
      auto raw_key = d_raw_key.get_access<sycl_read_write>(cgh);
      auto raw_nonce = d_raw_nonce.get_access<sycl_read_write>(cgh);
      auto raw_keystream = d_raw_keystream.get_access<sycl_read_write>(cgh);
      auto result = d_result.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class keystreams>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
       test_keystreams(
          item,
          key.get_pointer(),
          nonce.get_pointer(),
          keystream.get_pointer(),
          char2uint.get_pointer(), 
          raw_key.get_pointer(),
          raw_nonce.get_pointer(),
          raw_keystream.get_pointer(),
          result.get_pointer(),
          key_len, nonce_len, keystream_len);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels: %f (us)\n", (time * 1e-3f) / repeat);

  q.submit([&] (handler &cgh) {
    auto acc = d_raw_keystream.get_access<sycl_read>(cgh);
    cgh.copy(acc, raw_keystream);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_result.get_access<sycl_read>(cgh);
    cgh.copy(acc, result);
  });

  q.wait();

  int error = memcmp(result, raw_keystream, result_len);
  printf("%s\n", error ? "FAIL" : "PASS");

  free(result);
  free(raw_keystream);

  return 0;
}
