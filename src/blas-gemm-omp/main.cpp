// Adapted from Intel MKL example to use cuBLAS with OpenMP offload
// Original used Intel MKL with omp dispatch, this version uses cuBLAS

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <omp.h>
#include <cublas_v2.h>

template <typename T>
void print_2x2_matrix_values(T M, int ldM, std::string M_name)
{
  std::cout << std::endl;
  std::cout << "\t\t\t" << M_name << " = [ " << (float)M[0*ldM + 0] << ", " << (float)M[1*ldM + 0] << ", ...\n";
  std::cout << "\t\t\t    [ " << (float)M[0*ldM + 1] << ", " << (float)M[1*ldM + 1] << ", ...\n";
  std::cout << "\t\t\t    [ " << "...\n";
  std::cout << std::endl;
}

template <typename fp>
void rand_matrix(fp *M, int n_row, int n_col)
{
  for (int i = 0; i < n_row; i++)
    for (int j = 0; j < n_col; j++)
      M[i * n_col + j] = rand() % 2;
}

template <typename fp>
void run_gemm_example(int m, int k, int n, int repeat) {

  // set scalar fp values
  fp alpha = fp(2.0);
  fp beta  = fp(0.5);

  const size_t A_size = sizeof(fp) * m * k;
  const size_t B_size = sizeof(fp) * k * n;
  const size_t C_size = sizeof(fp) * m * n;

  // prepare matrix data on host
  fp* a = (fp *)malloc(A_size);
  fp* b = (fp *)malloc(B_size);
  fp* c = (fp *)malloc(C_size);

  srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);

  // Allocate and copy to device using OpenMP
  fp *d_a, *d_b, *d_c;

  #pragma omp target enter data map(alloc: a[0:m*k], b[0:k*n], c[0:m*n])
  #pragma omp target update to(a[0:m*k], b[0:k*n], c[0:m*n])

  // Get device pointers
  #pragma omp target data use_device_addr(a, b, c)
  {
    d_a = a;
    d_b = b;
    d_c = c;
  }

  // Create cuBLAS handle and perform GEMM
  cublasHandle_t handle;
  cublasCreate(&handle);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
  {
    if constexpr (std::is_same_v<fp, __half>) {
      cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                 (__half*)&alpha, (__half*)d_b, n,
                 (__half*)d_a, k, (__half*)&beta,
                 (__half*)d_c, n);
    }
    else if constexpr (std::is_same_v<fp, float>) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                 &alpha, d_b, n, d_a, k, &beta, d_c, n);
    }
    else if constexpr (std::is_same_v<fp, double>) {
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                 &alpha, d_b, n, d_a, k, &beta, d_c, n);
    }
  }

  cudaDeviceSynchronize();
  cublasDestroy(handle);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average cuBLAS GEMM execution time: %f (us)\n", (time * 1e-3f) / repeat);

  // Copy results back
  #pragma omp target update from(c[0:m*n])
  #pragma omp target exit data map(delete: a[0:m*k], b[0:k*n], c[0:m*n])

  std::cout << "\n\t\tOutputting 2x2 block of C matrix:" << std::endl;
  print_2x2_matrix_values(c, n, "C");

  free(a);
  free(b);
  free(c);
}

int main (int argc, char ** argv) {
  if (argc != 5) {
    printf("Usage: %s <m> <k> <n> <repeat>\n", argv[0]);
    return 1;
  }
  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int n = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  std::cout << "\n\tRunning with single precision real data type:" << std::endl;
  run_gemm_example<float>(m, k, n, repeat);

  std::cout << "\n\tRunning with double precision real data type:" << std::endl;
  run_gemm_example<double>(m, k, n, repeat);

  return 0;
}
