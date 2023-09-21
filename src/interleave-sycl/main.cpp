/*
 *  Extension to the interleaving example in CUDA Programming by Shane Cook
 */
#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>

#define NUM_ELEMENTS 4096
#define COUNT 4096       // accumulation count

// an interleaved type
typedef struct
{
  unsigned int s0;
  unsigned int s1;
  unsigned int s2;
  unsigned int s3;
  unsigned int s4;
  unsigned int s5;
  unsigned int s6;
  unsigned int s7;
  unsigned int s8;
  unsigned int s9;
  unsigned int sa;
  unsigned int sb;
  unsigned int sc;
  unsigned int sd;
  unsigned int se;
  unsigned int sf;
} INTERLEAVED_T;

// Define an array type based on the interleaved structure
typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

// Alternative - structure of arrays
typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct
{
  ARRAY_MEMBER_T s0;
  ARRAY_MEMBER_T s1;
  ARRAY_MEMBER_T s2;
  ARRAY_MEMBER_T s3;
  ARRAY_MEMBER_T s4;
  ARRAY_MEMBER_T s5;
  ARRAY_MEMBER_T s6;
  ARRAY_MEMBER_T s7;
  ARRAY_MEMBER_T s8;
  ARRAY_MEMBER_T s9;
  ARRAY_MEMBER_T sa;
  ARRAY_MEMBER_T sb;
  ARRAY_MEMBER_T sc;
  ARRAY_MEMBER_T sd;
  ARRAY_MEMBER_T se;
  ARRAY_MEMBER_T sf;
} NON_INTERLEAVED_T;

// data initialization and verification
#include "util.cpp"

void add_test_interleaved(
    sycl::queue &q,
    INTERLEAVED_T * const h_dst,
    const INTERLEAVED_T * const h_src,
    const int repeat,
    const unsigned int num_elements)
{
  // Set launch params
  const unsigned int local_work_size = 256;
  const unsigned int global_work_size = 
	  (num_elements + (local_work_size-1)) / local_work_size * local_work_size;

  INTERLEAVED_T *d_src = sycl::malloc_device<INTERLEAVED_T>(num_elements, q);
  INTERLEAVED_T *d_dst = sycl::malloc_device<INTERLEAVED_T>(num_elements, q);
  
  const size_t num_bytes = (sizeof(INTERLEAVED_T) * num_elements);
  q.memcpy(d_src, h_src, num_bytes);
  q.memcpy(d_dst, h_dst, num_bytes);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class interleaved>(
        sycl::nd_range<1>(global_work_size, local_work_size), [=](sycl::nd_item<1> item) {
        const unsigned int tid = item.get_global_id(0);
        if (tid < num_elements)
        {
          for (unsigned int i=0; i<COUNT; i++)
          {
            d_dst[tid].s0 += d_src[tid].s0;
            d_dst[tid].s1 += d_src[tid].s1;
            d_dst[tid].s2 += d_src[tid].s2;
            d_dst[tid].s3 += d_src[tid].s3;
            d_dst[tid].s4 += d_src[tid].s4;
            d_dst[tid].s5 += d_src[tid].s5;
            d_dst[tid].s6 += d_src[tid].s6;
            d_dst[tid].s7 += d_src[tid].s7;
            d_dst[tid].s8 += d_src[tid].s8;
            d_dst[tid].s9 += d_src[tid].s9;
            d_dst[tid].sa += d_src[tid].sa;
            d_dst[tid].sb += d_src[tid].sb;
            d_dst[tid].sc += d_src[tid].sc;
            d_dst[tid].sd += d_src[tid].sd;
            d_dst[tid].se += d_src[tid].se;
            d_dst[tid].sf += d_src[tid].sf;
          }
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (interleaved) execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_dst, d_dst, num_bytes).wait();
  sycl::free(d_src, q);
  sycl::free(d_dst, q);
}

void add_test_non_interleaved(
    sycl::queue &q,
    NON_INTERLEAVED_T * const h_dst,
    const NON_INTERLEAVED_T * const h_src,
    const int repeat,
    const unsigned int num_elements)
{
  // Set launch params
  const unsigned int local_work_size = 256;
  const unsigned int global_work_size = 
	  (num_elements + (local_work_size-1)) / local_work_size * local_work_size;

  // Allocate memory on the device
  NON_INTERLEAVED_T *d_src = sycl::malloc_device<NON_INTERLEAVED_T>(1, q);
  NON_INTERLEAVED_T *d_dst = sycl::malloc_device<NON_INTERLEAVED_T>(1, q);

  const size_t num_bytes = sizeof(NON_INTERLEAVED_T);
  q.memcpy(d_src, h_src, num_bytes);
  q.memcpy(d_dst, h_dst, num_bytes);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class non_interleaved>(
        sycl::nd_range<1>(global_work_size, local_work_size), [=](sycl::nd_item<1> item) {
        const unsigned int tid = item.get_global_id(0);
        if (tid < num_elements)
        {
          for (unsigned int i=0; i<COUNT; i++)
          {
            d_dst->s0[tid] += d_src->s0[tid];
            d_dst->s1[tid] += d_src->s1[tid];
            d_dst->s2[tid] += d_src->s2[tid];
            d_dst->s3[tid] += d_src->s3[tid];
            d_dst->s4[tid] += d_src->s4[tid];
            d_dst->s5[tid] += d_src->s5[tid];
            d_dst->s6[tid] += d_src->s6[tid];
            d_dst->s7[tid] += d_src->s7[tid];
            d_dst->s8[tid] += d_src->s8[tid];
            d_dst->s9[tid] += d_src->s9[tid];
            d_dst->sa[tid] += d_src->sa[tid];
            d_dst->sb[tid] += d_src->sb[tid];
            d_dst->sc[tid] += d_src->sc[tid];
            d_dst->sd[tid] += d_src->sd[tid];
            d_dst->se[tid] += d_src->se[tid];
            d_dst->sf[tid] += d_src->sf[tid];
          }
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (non-interleaved) execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_dst, d_dst, num_bytes).wait();
  sycl::free(d_src, q);
  sycl::free(d_dst, q);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  NON_INTERLEAVED_T non_interleaved_src, non_interleaved_dst; 
  INTERLEAVED_ARRAY_T interleaved_src, interleaved_dst; 

  initialize (interleaved_src, interleaved_dst,
              non_interleaved_src, non_interleaved_dst, NUM_ELEMENTS);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  add_test_non_interleaved(q, &non_interleaved_dst, &non_interleaved_src,
                           repeat, NUM_ELEMENTS);
  add_test_interleaved(q, interleaved_dst, interleaved_src,
                       repeat, NUM_ELEMENTS);
  verify(interleaved_dst, non_interleaved_dst, NUM_ELEMENTS);

  return 0;
}
