#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

int main (int argc, char* argv[]){
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  bool ok = true;
  cudaError_t cudaStat;
  cublasStatus_t cublasStat;
  cublasHandle_t handle;

  cublasStat = cublasCreate(&handle);
  if (cublasStat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
  }

  for (int n = 512*1024; n <= 1024*1024*512; n = n * 2) {
    int i, j;
    size_t size = n * sizeof(float);
    float* a = (float *)malloc (size);
    if (!a) {
      printf ("host memory allocation failed");
      break;
    }

    // reference
    double gold = 0.0;  // double is required
    for (i = 0; i < n; i++) {
      a[i] = (float)((i+1) % 7);
      gold += a[i]*a[i];
    }
    gold = sqrt(gold);

    long start = get_time();

    float* d_a;
    cudaStat = cudaMalloc ((void**)&d_a, size);
    if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
    }

    cudaStat = cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
      printf ("device memory copy failed");
    }

    float result;
    for (j = 0; j < repeat; j++)
      cublasStat = cublasSnrm2(handle, n, d_a, 1, &result);

    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS Snrm2 failed\n");
    }
    
    cudaStat = cudaFree(d_a);
    if (cudaStat != cudaSuccess) {
      printf ("device memory deallocation failed");
    }

    long end = get_time();
    printf("#elements = %.2f M, measured time = %.3f s\n", 
            n / (1024.f*1024.f), (end-start) / 1e6f);

    if (a != NULL) free(a);

    if (fabsf((float)gold - result) > 1e-3f) {
      printf("FAIL: %f %f #elements = %d\n", gold, result, i);
      ok = false;
      break;
    }
  }

  cublasDestroy(handle);

  if (ok) printf("PASS\n");
  return 0;
}
