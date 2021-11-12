#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

__global__ void reverse (int *d, const int len)
{
  __shared__ int s[256];
  int t = threadIdx.x;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[len-t-1];
}

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: ./%s <iterations>\n", argv[0]);
    return 1;
  }

  // specify the number of test cases
  const int iteration = atoi(argv[1]);

  // number of elements to reverse
  const int len = 256;
  const int elem_size = len * sizeof(int);

  // device result
  int test[len];

  // expected results after reverse operations even/odd times
  int error = 0;
  int gold_odd[len];
  int gold_even[len];

  for (int i = 0; i < len; i++) {
    gold_odd[i] = len-i-1;
    gold_even[i] = i;
  }

  int *d_test;
  hipMalloc((void**)&d_test, elem_size);

  srand(123);
  for (int i = 0; i < iteration; i++) {

    const int count = rand() % 10000 + 100;  // bound the reverse range

    hipMemcpy(d_test, gold_even, elem_size, hipMemcpyHostToDevice);

    for (int j = 0; j < count; j++)
      hipLaunchKernelGGL(reverse, dim3(1), dim3(len), 0, 0, d_test, len);

    hipMemcpy(test, d_test, elem_size, hipMemcpyDeviceToHost);

    if (count % 2 == 0)
      error = memcmp(test, gold_even, elem_size);
    else
      error = memcmp(test, gold_odd, elem_size);
    
    if (error) break;
  }
  
  printf("%s\n", error ? "FAIL" : "PASS");

  hipFree(d_test);
  return 0;
}
