/*
 *  Extension to the interleaving example in CUDA Programming by Shane Cook
 */
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define NUM_ELEMENTS 4096
#define REPEAT 4096       // accumulation count

// an interleaved type
typedef struct dpct_type_713de3
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
typedef struct dpct_type_ed2a2f
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

void add_kernel_interleaved(
    INTERLEAVED_T * const dest_ptr,
    const INTERLEAVED_T * const src_ptr,
    const unsigned int iter,
    const unsigned int num_elements,
    sycl::nd_item<3> item_ct1)
{
  const unsigned int tid =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);
  if (tid < num_elements)
  {
    for (unsigned int i=0; i<iter; i++)
    {
      dest_ptr[tid].s0 += src_ptr[tid].s0;
      dest_ptr[tid].s1 += src_ptr[tid].s1;
      dest_ptr[tid].s2 += src_ptr[tid].s2;
      dest_ptr[tid].s3 += src_ptr[tid].s3;
      dest_ptr[tid].s4 += src_ptr[tid].s4;
      dest_ptr[tid].s5 += src_ptr[tid].s5;
      dest_ptr[tid].s6 += src_ptr[tid].s6;
      dest_ptr[tid].s7 += src_ptr[tid].s7;
      dest_ptr[tid].s8 += src_ptr[tid].s8;
      dest_ptr[tid].s9 += src_ptr[tid].s9;
      dest_ptr[tid].sa += src_ptr[tid].sa;
      dest_ptr[tid].sb += src_ptr[tid].sb;
      dest_ptr[tid].sc += src_ptr[tid].sc;
      dest_ptr[tid].sd += src_ptr[tid].sd;
      dest_ptr[tid].se += src_ptr[tid].se;
      dest_ptr[tid].sf += src_ptr[tid].sf;
    }
  }
}

void add_kernel_non_interleaved(
    NON_INTERLEAVED_T * const dest_ptr,
    const NON_INTERLEAVED_T * const src_ptr,
    const unsigned int iter,
    const unsigned int num_elements,
    sycl::nd_item<3> item_ct1)
{
  const unsigned int tid =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);
  if (tid < num_elements)
  {
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
}

void add_test_interleaved(
    INTERLEAVED_T * const h_dst,
    const INTERLEAVED_T * const h_src,
    const unsigned int iter,
    const unsigned int num_elements)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  // Set launch params
  const unsigned int num_threads = 256;
  const unsigned int num_blocks = (num_elements + (num_threads-1)) / num_threads;
  // Allocate memory on the device
  const size_t num_bytes = (sizeof(INTERLEAVED_T) * num_elements);
  //printf("%lu\n", num_bytes);
  INTERLEAVED_T * d_dst;
  INTERLEAVED_T * d_src;
  d_src = (INTERLEAVED_T *)sycl::malloc_device(num_bytes, q_ct1);
  d_dst = (INTERLEAVED_T *)sycl::malloc_device(num_bytes, q_ct1);
  q_ct1.memcpy(d_src, h_src, num_bytes).wait();
  q_ct1.memcpy(d_dst, h_dst, num_bytes).wait();
  for (int n = 0; n < 100; n++)
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         add_kernel_interleaved(d_dst, d_src, iter,
                                                num_elements, item_ct1);
                       });
    });
  q_ct1.memcpy(h_dst, d_dst, num_bytes).wait();
  sycl::free(d_src, q_ct1);
  sycl::free(d_dst, q_ct1);
}

void add_test_non_interleaved(
    NON_INTERLEAVED_T * const h_dst,
    const NON_INTERLEAVED_T * const h_src,
    const unsigned int iter,
    const unsigned int num_elements)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  // Set launch params
  const unsigned int num_threads = 256;
  const unsigned int num_blocks = (num_elements + (num_threads-1)) / num_threads;
  // Allocate memory on the device
  const size_t num_bytes = sizeof(NON_INTERLEAVED_T);
  //printf("%lu\n", num_bytes);
  NON_INTERLEAVED_T * d_dst;
  NON_INTERLEAVED_T * d_src;
  d_src = (NON_INTERLEAVED_T *)sycl::malloc_device(num_bytes, q_ct1);
  d_dst = (NON_INTERLEAVED_T *)sycl::malloc_device(num_bytes, q_ct1);
  q_ct1.memcpy(d_src, h_src, num_bytes).wait();
  q_ct1.memcpy(d_dst, h_dst, num_bytes).wait();
  for (int n = 0; n < 100; n++)
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         add_kernel_non_interleaved(d_dst, d_src, iter,
                                                    num_elements, item_ct1);
                       });
    });
  q_ct1.memcpy(h_dst, d_dst, num_bytes).wait();
  sycl::free(d_src, q_ct1);
  sycl::free(d_dst, q_ct1);
}

int main() {
  NON_INTERLEAVED_T non_interleaved_src, non_interleaved_dst; 
  INTERLEAVED_ARRAY_T interleaved_src, interleaved_dst; 
  initialize (interleaved_src, interleaved_dst, 
		  non_interleaved_src, non_interleaved_dst, NUM_ELEMENTS);
  add_test_non_interleaved(&non_interleaved_dst, &non_interleaved_src, REPEAT, NUM_ELEMENTS);
  add_test_interleaved(interleaved_dst, interleaved_src, REPEAT, NUM_ELEMENTS);
  verify(interleaved_dst, non_interleaved_dst, NUM_ELEMENTS);
  return 0;
}
