#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <utility>
#include <oneapi/mkl.hpp>
#include "common.h"

#define uS_PER_SEC 1000000
#define uS_PER_mS 1000

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage %s <matrix col> <matrix row> <repeat times>\n", argv[0]);
    return 1;
  }

  int M = atoi(argv[1]); 
  int N = atoi(argv[2]); 
  const int repeat = atoi(argv[3]); 
  if (M <= 0 || N <= 0 || repeat < 0) {
    printf("Error: invalid inputs\n");
    return 1;
  }

  int error = 0;

#ifdef USE_GPU
  sycl::gpu_selector dev_sel;
#else
  sycl::cpu_selector dev_sel;
#endif
  sycl::queue q(dev_sel);

  const size_t size = N * M;
  const size_t size_byte = size * sizeof(float);

  // N rows and M columns
  float *matrix = (float *) malloc (size_byte);
  for (size_t i = 0; i < size; i++) matrix[i] = rand() % 13; 

  timeval t1, t2;
  gettimeofday(&t1, NULL);

  // M rows and N columns
  float *matrixT = (float *) malloc (size_byte);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      matrixT[(j*N)+i] = matrix[(i*M)+j];

  gettimeofday(&t2, NULL);
  float et1 = (((t2.tv_sec*uS_PER_SEC)+t2.tv_usec) - ((t1.tv_sec*uS_PER_SEC)+t1.tv_usec))/(float)uS_PER_mS;
  printf("CPU time = %fms\n", et1);

  const float alpha = 1.f;
  const float beta  = 0.f;

  // store host and device results
  float *h_matrixT , *d_matrixT , *d_matrix;
  h_matrixT = (float *) malloc (size_byte);
  d_matrixT = (float *)sycl::malloc_device(size_byte, q);
  d_matrix = (float *)sycl::malloc_device(size_byte, q);

  q.memcpy(d_matrix, matrix, size_byte).wait();

  std::vector<event> deps;

  // start the device timing
  gettimeofday(&t1, NULL);

  for (int i = 0; i < repeat; i++) {
    event status;
    try {
      status = oneapi::mkl::blas::omatadd_batch(
        q,
        oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans,
        N,
        M,
        alpha,
        d_matrix,
        M,
        size,
        beta,
        d_matrix,
        N,
        size,
        d_matrixT,
        N,
        size,
        1,
        deps);
    } catch(sycl::exception const& e) {
      std::cout << "\t\tCaught SYCL exception during omatadd_batch:\n"
                << e.what() << std::endl;
      error = 1;
      break;
    }
    status.wait();
    std::swap(d_matrix, d_matrixT);
    std::swap(N, M);
  }

  gettimeofday(&t2, NULL);

  float et2 = (((t2.tv_sec*uS_PER_SEC)+t2.tv_usec) - 
               ((t1.tv_sec*uS_PER_SEC)+t1.tv_usec)) / (float)uS_PER_mS;

  printf("Average device execution time = %fms\n", et2 / repeat);

  q.memcpy(h_matrixT, d_matrix, size_byte).wait();

  sycl::free(d_matrix, q);
  sycl::free(d_matrixT, q);

  if (error == 0) {

    // check host and device results
    if (repeat % 2)
      error = memcmp(h_matrixT, matrixT, size_byte);
    else
      error = memcmp(h_matrixT, matrix, size_byte);
  }

  printf("%s\n", error ? "FAIL" : "PASS");

  free(matrixT);
  free(matrix);
  return 0;
}
