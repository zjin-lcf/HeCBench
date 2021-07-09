#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shmem_kernels.h"

#define VECTOR_SIZE (1024*1024)

int main(int argc, char* argv[]) {

  const int n = atoi(argv[1]);  // launch kernel n times

  printf("shared memory bandwidth microbenchmark\n");

  unsigned int datasize = VECTOR_SIZE*sizeof(double);

  printf("Buffer sizes: %dMB\n", datasize/(1024*1024));

  double *c = (double*)malloc(datasize);
  memset(c, 0, sizeof(int)*VECTOR_SIZE);

  // benchmark execution
  shmembenchGPU(c, VECTOR_SIZE, n);

  free(c);

  return 0;
}

