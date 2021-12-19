
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "common.h"

// Copyright 2004-present Facebook. All Rights Reserved.
// Constructs a rounding factor used to truncate elements in a sum
// such that the sum of the truncated elements is the same no matter
// what the order of the sum is.
//
// Floating point summation is not associative; using this factor
// makes it associative, so a parallel sum can be performed in any
// order (presumably using atomics).
//
// Follows Algorithm 5: Reproducible Sequential Sum in
// 'Fast Reproducible Floating-Point Summation' by Demmel and Nguyen
// http://www.eecs.berkeley.edu/~hdnguyen/public/papers/ARITH21_Fast_Sum.pdf
//
// For summing x_i, i = 1 to n:
// @param max The maximum seen floating point value abs(x_i)
// @param n The number of elements for the sum, or an upper bound estimate
inline float
createRoundingFactor(float max, int n) {
  float delta = (max * (float)n) / (1.f - 2.f * (float)n * FLT_EPSILON);

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  sycl::frexp(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return sycl::ldexp(1.f, exp);
}
  
// Given the rounding factor in `createRoundingFactor` calculated
// using max(|x_i|), truncate `x` to a value that can be used for a
// deterministic, reproducible parallel sum of all x_i.
inline float
truncateWithRoundingFactor(float roundingFactor, float x) {
  return (roundingFactor + x) -  // rounded
         roundingFactor;         // exactly
}

void sumArray (
  nd_item<1> &item,
  const float factor, 
  const   int length,
  const float *__restrict__ x,
        float *__restrict__ r)
{
  for (int i = item.get_global_id(0); i < length; 
           i += item.get_local_range(0) * item.get_group_range(0)) {
    float q = truncateWithRoundingFactor(factor, x[i]);
    // atomicAdd(r, q);  // sum in any order
    auto ao = ext::oneapi::atomic_ref<float, 
              ext::oneapi::memory_order::relaxed,
              ext::oneapi::memory_scope::device,
              access::address_space::global_space> (r[0]);
    ao.fetch_add(q);
  }
}
  
void sumArrays (
  nd_item<1> &item,
  const int nArrays,
  const int length,
  const float *__restrict__ x,
        float *__restrict__ r,
  const float *__restrict__ maxVal)
{
  for (int i = item.get_global_id(0); i < nArrays; 
           i += item.get_local_range(0) * item.get_group_range(0)) {
    x += i * length;
    float factor = createRoundingFactor(maxVal[i], length);
    float s = 0;
    for (int n = length-1; n >= 0; n--)  // sum in reverse order
      s += truncateWithRoundingFactor(factor, x[n]);
    r[i] = s;
  }
}
  
int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of arrays> <length of each array>\n", argv[0]); 
    return 1;
  }

  const int nArrays = atoi(argv[1]);
  const int nElems = atoi(argv[2]);
  const size_t narray_size = sizeof(float) * nArrays;
  const size_t array_size = narray_size * nElems;

  // set of arrays
  float *arrays = (float*) malloc (array_size);
  // max value of each array 
  float *maxVal = (float*) malloc (narray_size);
  // sum of each array
  float *result = (float*) malloc (narray_size);
  // rounding factor of each array
  float *factor = (float*) malloc (narray_size);
  // reference sum of each array
  float *result_ref = (float*) malloc (narray_size);

  srand(123);

  // compute max value and rounding factor of each array on a host
  float *arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    float max = 0;
    for (int i = 0; i < nElems; i++) {
      arr[i] = (float)rand() / (float)RAND_MAX;
      if (rand() % 2) arr[i] = -1.f * arr[i];
      max = fmaxf(fabs(arr[i]), max);
    }
    factor[n] = createRoundingFactor(max, nElems);
    maxVal[n] = max;
    arr += nElems;
  }

  // compute the sum of each array on a host
  arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    result_ref[n] = 0;
    for (int i = 0; i < nElems; i++)
      result_ref[n] += truncateWithRoundingFactor(factor[n], arr[i]);
    arr += nElems;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_arrays (arrays, nArrays * nElems);
  buffer<float, 1> d_maxVal (maxVal, nArrays);
  buffer<float, 1> d_result (nArrays);

  range<1> gws (256*256);
  range<1> lws (256);
 
  // reset results for the sumArray kernel
  q.submit([&] (handler &cgh) {
    auto acc = d_result.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  for (int n = 0; n < nArrays; n++) {
    // sum over each array
    const float f = factor[n];
    q.submit([&] (handler &cgh) {
      auto arr = d_arrays.get_access<sycl_read>(cgh, range<1>(nElems), id<1>(n*nElems));
      auto out = d_result.get_access<sycl_read_write>(cgh, range<1>(1), id<1>(n));
      cgh.parallel_for<class sum_array>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        sumArray (item, f, nElems, arr.get_pointer(), out.get_pointer());
      });
    });
  }

  // bit accurate sum
  q.submit([&] (handler &cgh) {
    auto acc = d_result.get_access<sycl_read>(cgh);
    cgh.copy(acc, result);
  }).wait();

  bool ok = !memcmp(result_ref, result, narray_size);
  printf("%s\n", ok ? "PASS" : "FAIL");
  
  // sum over arrays
  q.submit([&] (handler &cgh) {
    auto arr = d_arrays.get_access<sycl_read>(cgh);
    auto out = d_result.get_access<sycl_discard_write>(cgh);
    auto max = d_maxVal.get_access<sycl_read>(cgh);
    cgh.parallel_for<class sum_arrays>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      sumArrays (item, nArrays, nElems, arr.get_pointer(), 
                 out.get_pointer(), max.get_pointer());
    });
  });

  // bit accurate sum
  q.submit([&] (handler &cgh) {
    auto acc = d_result.get_access<sycl_read>(cgh);
    cgh.copy(acc, result);
  }).wait();

  ok = !memcmp(result_ref, result, narray_size);
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(arrays);
  free(maxVal);
  free(result);
  free(factor);
  free(result_ref);

  return 0;
}
