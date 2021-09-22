#include <iostream>
#include <limits>
#include <cmath>

using namespace std;

/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int m_size = 768 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

#include "verify.cpp"

int main() {
  int i, j;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[P] = new float[N][P];
  // host-side cpu result
  float(*c_host)[P] = new float[M][P];
  // host-side gpu result
  float(*c_back)[P] = new float[M][P];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = 1.f / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < P; j++) { 
    float sum = 0;
    for (i = 0; i < N; i++)
      sum += b_host[i][j];
    for (i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.

  cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
       << ") * b(" << N << "," << P << ")\n";

  #pragma omp target teams distribute parallel for collapse(2) \
  map(to : a_host[0:M][0:N], b_host[0:N][0:P]) map(from : c_back[0:M][0:P]) 
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) {
        sum += sqrtf(a_host[i][k] * b_host[k][j]);
      }
      const float value = 1.f - sum;
      const float gate = (!signbit(value));
      c_back[i][j] = sqrtf(gate * value);
    }
  }

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  return 0;
}

