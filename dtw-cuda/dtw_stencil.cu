#include "dtw_stencil.cuh"

/** Take the softmin of 3 elements
 * @param a The first element
 * @param b The second element
 * @param c The third element
 * @param gamma The smoothing factor
 */
__device__
float softmin(float a, float b, float c, const float gamma)
{
  float ng = -gamma;
  a /= ng;
  b /= ng;
  c /= ng;
  float max_of = fmax(fmax(a, b), c);
  float sum = exp(a - max_of) + exp(b - max_of) + exp(c - max_of);

  return ng * (log(sum) + max_of);
}

/** Check whether i,j are within the Sakoe-Chiba band
 *  @param m The length of the first time series
 *  @param n The length of the second time series
 *  @param i The cell row index
 *  @param j The cell column index
 *  @param bandwidth Maximum warping distance from the diagonal to consider for
 *  optimal path calculation (Sakoe-Chiba band). 0 = unlimited.
 */
__device__ 
bool check_sakoe_chiba_band(int m, int n, int i, int j, int bandwidth)
{
  if (bandwidth == 0)
  {
    return true;
  }
  int width = abs(m - n) + bandwidth;
  int lower = max(1, (m > n ? j : i) - bandwidth);
  int upper = min(max(m, n), (m > n ? j : i) + width) + 1;
  bool is_in_lower = (m > n ? i : j) >= lower;
  bool is_in_upper = (m > n ? i : j) < upper;
  return is_in_lower && is_in_upper;
}

/** Kernel function for computing DTW on pairwise Euclidean
 * distance matrix for multivariate time series with CUDA.
 * Uses a shared memory stencil for caching the previous diagonal
 * Input D should be a __device__ array.
 * This naive version only works for sequence lengths <= 1024 i.e. can fit in
 * a single threadblock.
 * Each threadblock computes DTW for a pair of time series
 * Each thread can process one anti-diagonal.
 * @param D A 3D tensor of pairwise squared Euclidean distance matrices
 * between time series
 * @param R An m+2 x n+2 array that will be filled with the alignments
 * @param cost The total path costs will be written to this array of length nD
 * @param nD The number of distance matrices in D and its leading dimension
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma Smoothing parameter
 */
__global__ void dtw_stencil(float *D, float *R, float *cost, uint nD,
                            uint m, uint n, float gamma, uint bandwidth)
{
  // dynamic shared memory diagonal buffer array for caching the previous
  // diagonals.
  // length is (min(m,n) + 2) * 3 because it needs to store three
  // diagonals of R and the longest diagonal is (min(m,n)+2)
  // there should be min(m,n)+2 threads
  extern __shared__ float stencil[];
  const uint tx = threadIdx.x;
  const uint bx = blockIdx.x;

  uint bD = bx * m * n;
  uint bD2 = bx * (m + 2) * (n + 2);

  // number of antidiagonals is m+2+n+2-1
  const uint passes = m + n + 3;

  // each pass is one diagonal of the distance matrix
  for (uint p = 0; p < passes; p++)
  {
    uint pp = p;
    uint jj = max(0, min(pp - tx, n + 1));
    uint i = tx + 1;
    uint j = jj + 1;

    // calculate index offsets into the shared memory array for each
    // diagonal, using mod to rotate them.
    uint cur_idx = (pp + 2) % 3 * (blockDim.x);
    uint prev_idx = (pp + 1) % 3 * (blockDim.x);
    uint prev2_idx = pp % 3 * (blockDim.x);
    bool is_in_wave = tx + jj == pp && tx < m + 1 && jj < n + 1;
    bool is_in_band = check_sakoe_chiba_band(m + 1, n + 1, i, j, bandwidth);
    if (is_in_wave && is_in_band)
    {
      // load a diagonal into shared memory
      if (p == 0 && tx == 0)
      {
        stencil[prev2_idx] = 0;
      }
      stencil[prev2_idx + jj] = R[bD2 + tx * (n + 2) + jj];
    }
    // synchronize to make sure shared mem is done loading
    __syncthreads();

    bool is_in_D;
    pp = p - 2;
    jj = max(0, min(pp - tx, n));
    i = tx + 1;
    j = jj + 1;
    cur_idx = (pp + 2) % 3 * (blockDim.x);
    prev_idx = (pp + 1) % 3 * (blockDim.x);
    prev2_idx = pp % 3 * (blockDim.x);
    // check if this thread is on the current diagonal and in-bounds
    is_in_wave = tx + jj == pp && (tx < m + 1 && jj < n + 1);
    is_in_band = check_sakoe_chiba_band(m + 1, n + 1, i, j, bandwidth);
    is_in_D = (tx < m && jj < n);

    if (is_in_wave && is_in_band && is_in_D)
    {
      float c = D[bD + tx * n + jj];
      float r1 = stencil[prev_idx + i];
      float r2 = stencil[prev_idx + i - 1];
      float r3 = stencil[prev2_idx + i - 1];
      double prev_min = softmin(r1, r2, r3, gamma);
      stencil[cur_idx + i] = c + prev_min;
    }
    // make sure the diagonal is finished before proceeding to the next
    __syncthreads();

    // after a diagonal is no longer used, write that portion of R in
    // shared memory back to global memory
    if (is_in_wave && is_in_band)
    {
      R[bD2 + tx * (n + 2) + jj] = stencil[prev2_idx + tx];
    }
    // R[m,n] is the best path total cost, the last thread should
    // write this from the stencil back to the cost array in global memory
    if (p == passes - 1 && tx + jj == pp && tx < m + 1 && jj < n + 1)
    {
      cost[bx] = stencil[prev2_idx + tx];
    }
  }
}
