// B and C are vectors
template <typename T>
void gemmBatched_ref (int batchCount,
                      const int M_upper, const int K_upper, const int N_upper,
                      const int M, const int K, const int N,
                      const T alpha, const T beta,
                      const T* A, int lda, T* B, int ldb, T* C, int ldc)
{
  for (int p = 0; p < batchCount; ++p) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        T c_mnp = 0;
        for (int k = 0; k < K; ++k)
          c_mnp += A[p * M_upper * K_upper + m + k*lda] * B[p * K_upper * N_upper + k + n*ldb];
        C[p * M_upper * N_upper + m + n*ldc] = alpha*c_mnp + beta*C[p * M_upper * N_upper + m + n*ldc];
      }
    }
  }
}

