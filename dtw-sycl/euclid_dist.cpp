/** Kernels for use in computing squared euclidean distance matrix
 * @file euclid_dist.cu
 */

#include <cstdlib>
/** Compute the squared euclidean norm of matrix X
 *  @param m Height (rows) of matrix X
 *  @param k Width (columns) of matrix X
 *  @param XX a length m vector for the result
 */

void h_sq_euclid_norm(const unsigned int m, const unsigned int k, const float *X,
    float *XX)
{
  for (unsigned int i = 0; i < m; i++)
  {
    for (unsigned int j = 0; j < k; j++)
    {
      float x = X[i * k + j];
      XX[i] += x * x;
    }
  }
}

/* Compute the euclidean distance between two Euclidean norm
 * vectors XX and YY, i.e. X*X + Y*Y - 2X*Y
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param XX Squared Euclidean norm of X
 *  @param YY Squared Euclidean norm of Y
 *  @param XY 2 * X * Y^T (matrix multiplication result)
 *  @param D The result euclidean distance matrix with dimensions (m x n)
 */
void h_euclid_dist(const unsigned int m, const unsigned int n, const float *XX,
    const float *YY, const float *XY, float *D)
{
  for (unsigned int i = 0; i < m; i++)
  {
    for (unsigned int j = 0; j < n; j++)
    {
      D[i * n + j] = XX[i] + YY[j] - (XY[i * n + j]);
    }
  }
}

/* Single-precision matrix(MxK) matrix(KxN) multiply
 */
void sgemm (const unsigned int M, const unsigned int K, const unsigned int N, 
    const float *X, const float *Y, float *Z)
{
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      float s = 0.f;
      for (unsigned int k = 0; k < K; k++)
        s += 2.f * X[i*K+k] * Y[k*N+j];
      Z[i*N+j] = s;
    }
  }
}

void h_sq_euclid_dist_multi(const float *X, const float *Y, float *D,
    const unsigned int nX, const unsigned int nY, const unsigned int m,
    const unsigned int n, const unsigned int k)
{
  float *XX; // nX x m
  float *YY; // nY x n
  float *XY; // (nX x nY) x m x n

  XX = (float*) calloc (nX * m, sizeof(float));
  YY = (float*) calloc (nY * n, sizeof(float));
  XY = (float*) calloc (nX * m * nY *n, sizeof(float));

  // compute squared euclidean norm of X
  for (unsigned int i = 0; i < nX; i++)
  {
    h_sq_euclid_norm(m, k, &X[i * (m * k)], &XX[i * m]);
  }
  for (unsigned int i = 0; i < nY; i++)
  {
    h_sq_euclid_norm(n, k, &Y[i * (n * k)], &YY[i * n]);
  }

  // Compute 2*X*Y^T for each X and Y
  for (unsigned int i = 0; i < nX; i++)
  {
    for (unsigned int j = 0; j < nY; j++)
    {
      sgemm(m, k, n, &X[i*m*k], &Y[j*n*k], &XY[(i*nY+j)*m*n]);

      // compute XX + YY - 2XY for each pair of X and Y
      h_euclid_dist(
          m, n, &XX[i * m], &YY[j * n], &XY[(i * nY + j) * m * n],
          &D[(i * nY + j) * m * n]);
    }
  }
  free(XX);
  free(YY);
  free(XY);
}

