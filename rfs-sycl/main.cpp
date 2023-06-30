#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

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
  sycl::nd_item<1> &item,
  const float factor,
  const   int length,
  const float *__restrict__ x,
        float *__restrict__ r)
{
  for (int i = item.get_global_id(0); i < length;
           i += item.get_local_range(0) * item.get_group_range(0)) {
    float q = truncateWithRoundingFactor(factor, x[i]);
    // atomicAdd(r, q);  // sum in any order
    auto ao = sycl::atomic_ref<float,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (r[0]);
    ao.fetch_add(q);
  }
}

void sumArrays (
  sycl::nd_item<1> &item,
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
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_arrays = sycl::malloc_device<float>(nArrays * nElems, q);
  q.memcpy(d_arrays, arrays, array_size);

  float *d_maxVal = sycl::malloc_device<float>(nArrays, q);
  q.memcpy(d_maxVal, maxVal, narray_size);

  float *d_result = sycl::malloc_device<float>(nArrays, q);

  sycl::range<1> gws (256*256);
  sycl::range<1> lws (256);

  // reset results for the sumArray kernel
  q.memset(d_result, 0, narray_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < nArrays; n++) {
    // sum over each array
    const float f = factor[n];
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class sum_array>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        sumArray (item, f, nElems, d_arrays + n * nElems, d_result + n);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (sumArray): %f (s)\n", (time * 1e-9f) / nArrays);

  // bit accurate sum
  q.memcpy(result, d_result, narray_size).wait();

  bool ok = !memcmp(result_ref, result, narray_size);
  printf("%s\n", ok ? "PASS" : "FAIL");

  start = std::chrono::steady_clock::now();

  // sum over arrays
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class sum_arrays>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      sumArrays (item, nArrays, nElems, d_arrays, d_result, d_maxVal);
    });
  }).wait();

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (sumArrays): %f (s)\n", time * 1e-9f);

  // bit accurate sum
  q.memcpy(result, d_result, narray_size).wait();

  ok = !memcmp(result_ref, result, narray_size);
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_arrays, q);
  sycl::free(d_maxVal, q);
  sycl::free(d_result, q);
  free(arrays);
  free(maxVal);
  free(result);
  free(factor);
  free(result_ref);

  return 0;
}
