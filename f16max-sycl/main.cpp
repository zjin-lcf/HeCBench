#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"

#define NUM_OF_BLOCKS 1024
#define NUM_OF_THREADS 256

/*==================================================
 References
 https://x.momo86.net/en?p=113
 https://github.com/cpc/hipcl
 ==================================================*/

typedef struct __attribute__((__aligned__(4)))
{
  union {
    unsigned char c[4];
    unsigned int ui;
  };
} ucharHolder;

typedef struct __attribute__((__aligned__(8)))
{
  union {
    unsigned int ui[2];
    unsigned char c[8];
  };
} uchar2Holder;

inline
unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
  uchar2Holder cHoldVal;
  ucharHolder cHoldOut;
  cHoldVal.ui[0] = x;
  cHoldVal.ui[1] = y;
  cHoldOut.c[0] = cHoldVal.c[s & 0x7];
  cHoldOut.c[1] = cHoldVal.c[(s >> 4) & 0x7];
  cHoldOut.c[2] = cHoldVal.c[(s >> 8) & 0x7];
  cHoldOut.c[3] = cHoldVal.c[(s >> 12) & 0x7];
  return cHoldOut.ui;
}

half2 half_max(const half2 a, const half2 b) {
  const half2 sub = a - b;
  const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
  const unsigned sw = 0x00003210 | (((sign >> 21) | (sign >> 13)) * 0x11);
  const unsigned int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a), 
      *reinterpret_cast<const unsigned*>(&b), sw);
  return *reinterpret_cast<const half2*>(&res);
}

half half_max(const half a, const half b) {
  const half sub = a - b;
  const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
  const unsigned sw = 0x00000010 | ((sign >> 13) * 0x11);
  const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a), 
      *reinterpret_cast<const short*>(&b), sw);
  return *reinterpret_cast<const half*>(&res);
}

template <typename T>
void hmax(nd_item<1> &item,
          T const *__restrict const a,
          T const *__restrict const b,
          T *__restrict const r,
          const size_t size)
{
  for (size_t i = item.get_global_id(0);
              i < size; i += item.get_local_range(0) * item.get_group_range(0))
    r[i] = half_max(a[i], b[i]);
}

void generateInput(half2 * a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    half2 temp;
    temp.x() = static_cast<float>(rand() % 922021);
    temp.y() = static_cast<float>(rand() % 922021);
    a[i] = temp;
  }
}

// compute the maximum of two values
int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS*16;

  half2 * a, *b, *r;

  a = (half2*) malloc (size*sizeof(half2));
  b = (half2*) malloc (size*sizeof(half2));
  r = (half2*) malloc (size*sizeof(half2));

  // initialize input values
  srand(123); 
  generateInput(a, size);
  generateInput(b, size);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<half2, 1> d_a (a, size);
  buffer<half2, 1> d_b (b, size);
  buffer<half2, 1> d_r (size);

  range<1> gws (NUM_OF_BLOCKS * NUM_OF_THREADS);
  range<1> lws (NUM_OF_THREADS);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  // run hmax2
  for (int i = 0; i < repeat; i++)
    q.submit([&] (handler &cgh) {
      auto a = d_a.get_access<sycl_read>(cgh);
      auto b = d_b.get_access<sycl_read>(cgh);
      auto r = d_r.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class f16_max_v2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        hmax<half2>(item, a.get_pointer(), b.get_pointer(), r.get_pointer(), size);
      });
    });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (max_half2) execution time %f (s)\n", (time * 1e-9f) / repeat);

  // verify
  q.submit([&] (handler &cgh) {
    auto acc = d_r.get_access<sycl_read>(cgh);
    cgh.copy(acc, r);
  }).wait(); 

  bool ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    float2 fa = a[i].convert<float, sycl::rounding_mode::automatic>();
    float2 fb = b[i].convert<float, sycl::rounding_mode::automatic>(); 
    float2 fr = r[i].convert<float, sycl::rounding_mode::automatic>();
    float x = fmaxf(fa.x(), fb.x());
    float y = fmaxf(fa.y(), fb.y());
    if (fabsf(fr.x() - x) > 1e-3 || fabsf(fr.y() - y) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("fp16_hmax2 %s\n", ok ?  "PASS" : "FAIL");

  // run hmax (the size is doubled)
  auto d_a_re = d_a.reinterpret<half>(range<1>(2*size));
  auto d_b_re = d_b.reinterpret<half>(range<1>(2*size));
  auto d_r_re = d_r.reinterpret<half>(range<1>(2*size));

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    q.submit([&] (handler &cgh) {
      auto a = d_a_re.get_access<sycl_read>(cgh);
      auto b = d_b_re.get_access<sycl_read>(cgh);
      auto r = d_r_re.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class f16_max>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        hmax<half>(item, a.get_pointer(), b.get_pointer(), r.get_pointer(), size*2);
      });
    });

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (max_half) execution time %f (s)\n", (time * 1e-9f) / repeat);

  // verify
  q.submit([&] (handler &cgh) {
    auto acc = d_r.get_access<sycl_read>(cgh);
    cgh.copy(acc, r);
  }).wait(); 

  ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    float2 fa = a[i].convert<float, sycl::rounding_mode::automatic>();
    float2 fb = b[i].convert<float, sycl::rounding_mode::automatic>(); 
    float2 fr = r[i].convert<float, sycl::rounding_mode::automatic>();
    float x = fmaxf(fa.x(), fb.x());
    float y = fmaxf(fa.y(), fb.y());
    if (fabsf(fr.x() - x) > 1e-3 || fabsf(fr.y() - y) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("fp16_hmax %s\n", ok ?  "PASS" : "FAIL");

  free(a);
  free(b);
  free(r);

  return EXIT_SUCCESS;
}
