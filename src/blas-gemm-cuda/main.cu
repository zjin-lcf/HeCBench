#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <vector>
#include <type_traits>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

template <typename T>
void print_2x2_matrix_values(T M, int ldM, std::string M_name)
{
  std::cout << std::endl;
  std::cout << "\t\t\t" << M_name << " = [ " << (float)M[0*ldM + 0] << ", " << (float)M[1*ldM + 0]         << ", ...\n";
  std::cout << "\t\t\t    [ "                << (float)M[0*ldM + 1] << ", " << (float)M[1*ldM + 1] << ", ...\n";
  std::cout << "\t\t\t    [ "                << "...\n";
  std::cout << std::endl;
}

//
// helpers for initializing templated scalar data type values.
//
template <typename fp> void rand_matrix(fp *M, int n_row, int n_col)
{
  for (int i = 0; i < n_row; i++)
    for (int j = 0; j < n_col; j++)
      M[i * n_col + j] = rand() % 2;
}

//
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
template <typename fp>
void run_gemm_example(int m, int k, int n, int repeat) {

  //
  // Initialize data for Gemm
  //
  // C = alpha * op(A) * op(B)  + beta * C
  //

  // set scalar fp values
  const fp alpha = fp(2.0);
  const fp beta  = fp(0.5);

  const size_t A_size = sizeof(fp) * m * k;
  const size_t B_size = sizeof(fp) * k * n;
  const size_t C_size = sizeof(fp) * m * n;

  // prepare matrix data
  fp* a = (fp *) aligned_alloc(64, A_size);
  fp* b = (fp *) aligned_alloc(64, B_size);
  fp* c = (fp *) aligned_alloc(64, C_size);

  srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);

  fp *da, *db, *dc;
  cudaMalloc((void**)&da, A_size);
  cudaMalloc((void**)&db, B_size);
  cudaMalloc((void**)&dc, C_size);
  cudaMemcpy(da, a, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, B_size, cudaMemcpyHostToDevice);

  // create execution queue and buffers of matrix data
  cublasHandle_t h;
  cublasCreate(&h);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    if constexpr (std::is_same_v<fp, __half>)
      cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                  &alpha, db, n, da, k, &beta, dc, n);
    else if constexpr (std::is_same_v<fp, float>)
      cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                  &alpha, db, n, da, k, &beta, dc, n);
    else if constexpr (std::is_same_v<fp, double>)
      cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                  &alpha, db, n, da, k, &beta, dc, n);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average GEMM execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(c, dc, C_size, cudaMemcpyDeviceToHost);
  cublasDestroy(h);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  //
  // Post Processing
  //

  std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

  // output the top 2x2 block of A matrix
  //print_2x2_matrix_values(a, k, "A");

  // output the top 2x2 block of B matrix
  //print_2x2_matrix_values(b, n, "B");

  // output the top 2x2 block of C matrix
  print_2x2_matrix_values(c, n, "C");

  free(a);
  free(b);
  free(c);
}

//
// Main entry point for example.
//
int main (int argc, char ** argv) {
  if (argc != 5) {
    printf("Usage: %s <m> <k> <n> <repeat>\n", argv[0]);
    return 1;
  }
  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int n = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  std::cout << "\tRunning with half precision data type:" << std::endl;
  run_gemm_example<__half>(m, k, n, repeat);

  std::cout << "\tRunning with single precision data type:" << std::endl;
  run_gemm_example<float>(m, k, n, repeat);

  std::cout << "\tRunning with double precision data type:" << std::endl;
  run_gemm_example<double>(m, k, n, repeat);

  return 0;
}
