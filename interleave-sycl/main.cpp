/*
 *  Extension to the interleaving example in CUDA Programming by Shane Cook
 */
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define NUM_ELEMENTS 4096
#define REPEAT 4096       // accumulation count


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
    queue &q,
    INTERLEAVED_T * const h_dst,
    const INTERLEAVED_T * const h_src,
    const unsigned int iter,
    const unsigned int num_elements)
{
  // Set launch params
  const unsigned int local_work_size = 256;
  const unsigned int global_work_size = 
	  (num_elements + (local_work_size-1)) / local_work_size * local_work_size;

  buffer<INTERLEAVED_T, 1> d_src(h_src, num_elements);
  buffer<INTERLEAVED_T, 1> d_dst(h_dst, num_elements);

  for (int n = 0; n < 100; n++) {
    q.submit([&](handler &h) {
      auto src_acc = d_src.get_access<sycl_read>(h);
      auto dest_acc = d_dst.get_access<sycl_write>(h);
      h.parallel_for(nd_range<1>(global_work_size, local_work_size), [=](nd_item<1> item) {
        const unsigned int tid = item.get_global_id(0);
        if (tid < num_elements)
        {
          for (unsigned int i=0; i<iter; i++)
          {
            dest_acc[tid].s0 += src_acc[tid].s0;
            dest_acc[tid].s1 += src_acc[tid].s1;
            dest_acc[tid].s2 += src_acc[tid].s2;
            dest_acc[tid].s3 += src_acc[tid].s3;
            dest_acc[tid].s4 += src_acc[tid].s4;
            dest_acc[tid].s5 += src_acc[tid].s5;
            dest_acc[tid].s6 += src_acc[tid].s6;
            dest_acc[tid].s7 += src_acc[tid].s7;
            dest_acc[tid].s8 += src_acc[tid].s8;
            dest_acc[tid].s9 += src_acc[tid].s9;
            dest_acc[tid].sa += src_acc[tid].sa;
            dest_acc[tid].sb += src_acc[tid].sb;
            dest_acc[tid].sc += src_acc[tid].sc;
            dest_acc[tid].sd += src_acc[tid].sd;
            dest_acc[tid].se += src_acc[tid].se;
            dest_acc[tid].sf += src_acc[tid].sf;
          }
        }
      });
    });
  }
  q.wait();
}

void add_test_non_interleaved(
    queue &q,
    NON_INTERLEAVED_T * const h_dst,
    const NON_INTERLEAVED_T * const h_src,
    const unsigned int iter,
    const unsigned int num_elements)
{
  // Set launch params
  const unsigned int local_work_size = 256;
  const unsigned int global_work_size = 
	  (num_elements + (local_work_size-1)) / local_work_size * local_work_size;

  // Allocate memory on the device
  buffer<NON_INTERLEAVED_T, 1> d_src(h_src, 1);
  buffer<NON_INTERLEAVED_T, 1> d_dst(h_dst, 1);

  for (int n = 0; n < 100; n++) {
    q.submit([&](handler &h) {
      auto src_acc = d_src.get_access<sycl_read>(h);
      auto dest_acc = d_dst.get_access<sycl_write>(h);
      h.parallel_for(nd_range<1>(global_work_size, local_work_size), [=](nd_item<1> item) {
        const unsigned int tid = item.get_global_id(0);
        if (tid < num_elements)
        {
	  auto src_ptr = src_acc.get_pointer();
	  auto dest_ptr = dest_acc.get_pointer();
          for (unsigned int i=0; i<iter; i++)
          {
            dest_ptr->s0[tid] += src_ptr->s0[tid];
            dest_ptr->s1[tid] += src_ptr->s1[tid];
            dest_ptr->s2[tid] += src_ptr->s2[tid];
            dest_ptr->s3[tid] += src_ptr->s3[tid];
            dest_ptr->s4[tid] += src_ptr->s4[tid];
            dest_ptr->s5[tid] += src_ptr->s5[tid];
            dest_ptr->s6[tid] += src_ptr->s6[tid];
            dest_ptr->s7[tid] += src_ptr->s7[tid];
            dest_ptr->s8[tid] += src_ptr->s8[tid];
            dest_ptr->s9[tid] += src_ptr->s9[tid];
            dest_ptr->sa[tid] += src_ptr->sa[tid];
            dest_ptr->sb[tid] += src_ptr->sb[tid];
            dest_ptr->sc[tid] += src_ptr->sc[tid];
            dest_ptr->sd[tid] += src_ptr->sd[tid];
            dest_ptr->se[tid] += src_ptr->se[tid];
            dest_ptr->sf[tid] += src_ptr->sf[tid];
          }
        }
      });
    });
  }
  q.wait();
}

int main() {
  srand(2);

  NON_INTERLEAVED_T non_interleaved_src, non_interleaved_dst; 
  INTERLEAVED_ARRAY_T interleaved_src, interleaved_dst; 

  initialize (interleaved_src, interleaved_dst, 
		  non_interleaved_src, non_interleaved_dst, NUM_ELEMENTS);

#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  add_test_non_interleaved(q, &non_interleaved_dst, &non_interleaved_src, REPEAT, NUM_ELEMENTS);
  add_test_interleaved(q, interleaved_dst, interleaved_src, REPEAT, NUM_ELEMENTS);
  verify(interleaved_dst, non_interleaved_dst, NUM_ELEMENTS);

  return 0;
}
