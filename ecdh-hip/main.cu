#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "ecdh.h"

#define P_x 5
#define P_y 1
#define MODULUS 17
#define A 2

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <positive number of keys>\n", argv[0]);
    return 1;
  }

  // timing fast and slow versions for making public key
  struct timeval start_fast, end_fast, start_slow, end_slow;

  const int num_pk = atoi(argv[1]); 
  const int pk_x_size = num_pk * sizeof(int);
  const int pk_y_size = num_pk * sizeof(int);

  int *pk_x = (int*) malloc (pk_x_size);
  int *pk_y = (int*) malloc (pk_y_size);

  int *d_pk_x, *d_pk_y;
  hipMalloc((void**)&d_pk_x, pk_x_size);
  hipMalloc((void**)&d_pk_y, pk_y_size);
  
  dim3 grids = (num_pk + 255) / 256;

  gettimeofday(&start_slow,NULL);
  hipLaunchKernelGGL(k_slow, grids, 256, 0, 0, 18, P_x, P_y, d_pk_x, d_pk_y, MODULUS, A, num_pk);  
  hipMemcpy(pk_x, d_pk_x, pk_x_size, hipMemcpyDeviceToHost);
  hipMemcpy(pk_y, d_pk_y, pk_y_size, hipMemcpyDeviceToHost);
  gettimeofday(&end_slow,NULL);

  double elapsed_slow = (((end_slow.tv_sec*1000000.0 + end_slow.tv_usec) -
                        (start_slow.tv_sec*1000000.0 + start_slow.tv_usec)) / 1000000.00);

  printf("Public Key 0: (%u, %u), Time (slow): %f s\n",
         pk_x[0], pk_y[0], elapsed_slow);

  gettimeofday(&start_fast,NULL);
  hipLaunchKernelGGL(k_fast, grids, 256, 0, 0, 18, P_x, P_y, d_pk_x, d_pk_y, MODULUS, A, num_pk);  
  hipMemcpy(pk_x, d_pk_x, pk_x_size, hipMemcpyDeviceToHost);
  hipMemcpy(pk_y, d_pk_y, pk_y_size, hipMemcpyDeviceToHost);
  gettimeofday(&end_fast,NULL);
  double elapsed_fast = (((end_fast.tv_sec*1000000.0 + end_fast.tv_usec) -
                        (start_fast.tv_sec*1000000.0 + start_fast.tv_usec)) / 1000000.00);

  printf("Public Key 0: (%u, %u), Time (fast): %f s\n",
         pk_x[0], pk_y[0], elapsed_fast);
  
  hipFree(d_pk_x);
  hipFree(d_pk_y);
  free(pk_x);
  free(pk_y);
  return 0;
}
