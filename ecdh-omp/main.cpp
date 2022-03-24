#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include "ecdh.h"

#define P_x 5
#define P_y 1
#define MODULUS 17
#define a 2

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

  #pragma omp target data map(alloc: pk_x[0:num_pk], pk_y[0:num_pk])
  {
    gettimeofday(&start_slow,NULL);
    k_slow (18, P_x, P_y, pk_x, pk_y, MODULUS, a, num_pk);  
    #pragma omp target update from (pk_x[0:num_pk])
    #pragma omp target update from (pk_y[0:num_pk])
    gettimeofday(&end_slow,NULL);

    double elapsed_slow = (((end_slow.tv_sec*1000000.0 + end_slow.tv_usec) -
                          (start_slow.tv_sec*1000000.0 + start_slow.tv_usec)) / 1000000.00);

    printf("Public Key 0: (%u, %u), Time (slow): %f s\n",
           pk_x[0], pk_y[0], elapsed_slow);

    gettimeofday(&start_fast,NULL);
    k_fast(18, P_x, P_y, pk_x, pk_y, MODULUS, a, num_pk);  
    #pragma omp target update from (pk_x[0:num_pk])
    #pragma omp target update from (pk_y[0:num_pk])
    gettimeofday(&end_fast,NULL);
    double elapsed_fast = (((end_fast.tv_sec*1000000.0 + end_fast.tv_usec) -
                          (start_fast.tv_sec*1000000.0 + start_fast.tv_usec)) / 1000000.00);

    printf("Public Key 0: (%u, %u), Time (fast): %f s\n",
           pk_x[0], pk_y[0], elapsed_fast);
  }
  free(pk_x);
  free(pk_y);
  return 0;
}
