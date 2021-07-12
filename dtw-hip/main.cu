#include <chrono>
#include <cstring>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>

#include "euclid_dist.cuh"
#include "helper_functions.cuh"
#include "dtw_stencil.cuh"

using namespace std::chrono;

/** Host function to record the performance of each kernel
 *  on a given dataset
 *  @param X A vector of time series dataset with lenght m
 *  @param time_series_lenght The length of each series in X
 *  @param count The number of time series in Dataset

 * for multivariate time series with CUDA.
 * Input D should be a __device__ array of dimension (nD x m x n).
 * Each threadblock computes DTW for a pair of time series
 * m and n must each be no longer than 1024.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An (nD x (m+2) x (n+2)) device array to fill with alignment values.
 * @param costs A length nD array that will be filled with the pairwise costs
 * @param nD The number of distance matrices in D and its leading dimension
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 * @param bandwidth Maximum warping distance from the diagonal to consider for
 * optimal path calculation (Sakoe-Chiba band). Default = 0 = unlimited.
 */
__host__ void comparison(std::vector<float> &X, int time_series_len, int count)
{
  // Soft-DTW smoothing param
  float gamma = 0.1;
  // univariate time series
  const uint k = 1;
  // compare the same batch to itself
  const uint m = time_series_len;
  const uint n = time_series_len;
  const uint nX = count;
  const uint nY = count;
  const uint nD = nX * nY;
  size_t m2n2 = nD * (m + 2) * (n + 2);

  assert(min(m, n) <= 1024);

  // compute Euclidean distance matrix on a host
  float *hX = (float*) malloc (m * k * nX * sizeof(float));
  float *hY = (float*) malloc (n * k * nY * sizeof(float));
  float *hD = (float*) calloc (m * n * nD, sizeof(float));
  memcpy(hX, &X[0], m * k * nX * sizeof(float));
  memcpy(hY, &X[0], n * k * nY * sizeof(float));
  h_sq_euclid_dist_multi(hX, hY, hD, nX, nY, m, n, k);

  // const matrix 
  float *cost = (float*) malloc (nD * sizeof(float));

  // distance matrix 
  float *dD;
  cudaErrchk(hipMalloc(&dD, m * n * nD * sizeof(float)));
  cudaErrchk(hipMemcpy(dD, hD, m * n * nD * sizeof(float), hipMemcpyHostToDevice));

  // alignment matrix 
  float *dR;
  size_t sz_R = m2n2 * sizeof(float);
  cudaErrchk(hipMalloc((void**)&dR, sz_R));

  float *d_cost;
  cudaErrchk(hipMalloc((void**)&d_cost, nD * sizeof(float)));

  // The bandwidth is a tunable parameter (a percentage of the matrix dimension)
  for (int i = 1; i <= 5; i++) { 

    uint bandwidth = floor(i * 0.2 * m);
    auto start = high_resolution_clock::now();

    // fill matrix R with infinity
    dim3 inf_tpb(256);
    dim3 inf_blocks((m2n2 + 256 - 1) / 256);

    hipLaunchKernelGGL(fill_matrix_inf, dim3(inf_blocks), dim3(inf_tpb), 0, 0, 
        dR, (m + 2) * (n + 2), nD, std::numeric_limits<float>::infinity());

    cudaErrchk( hipPeekAtLastError() );

    uint threads = min(m, n) + 2;
    dim3 BLK (nD);
    dim3 TPB (threads);
    uint SMEM = threads * 3 * sizeof(float);

    hipLaunchKernelGGL(dtw_stencil, dim3(BLK), dim3(TPB), SMEM, 0, dD, dR, d_cost, nD, m, n, gamma, bandwidth);
    cudaErrchk( hipPeekAtLastError() );

    cudaErrchk(hipMemcpy(cost, d_cost, nD * sizeof(float), hipMemcpyDeviceToHost));

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << "dtw_stencil " << m << " " << nX << " " << duration << std::endl;

#ifdef DUMP
    for (uint j = 0; j < nD; j++) 
      std::cout << "bandwidth " << bandwidth  << " : "
                << std::setprecision(3) << cost[j] << std::endl;
    std::cout << std::endl;
#endif
  }

  hipFree(d_cost);
  hipFree(dR);
  hipFree(dD);
  free(cost);
  free(hX);
  free(hY);
}

/** Fill a vector with n random floats drawn from unit normal distribution.
 */
void fill_random(std::vector<float> &vec, uint n)
{
  std::default_random_engine gen(2);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (uint i = 0; i < n; i++)
  {
    vec.push_back(dist(gen));
  }
}

// To run as an example:
// make build
// ./bin/soft_dtw_perf data/ECG200/ECG200_TRAIN.txt
// output/ECG200/PERFORMANCE.CSV
int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0]
      << " [INPUT_FILENAME] | random [length] [count]\n";
    return 1;
  }

  std::vector<float> data_vec;
  std::string filename = argv[1];
  uint m = 0; // length of time series
  uint n = 0; // number of time series

  if (filename == "random")
  {
    if (argc < 4)
    {
      std::cerr << "Usage: " << argv[0] << " random [length] [count]\n";
      return 1;
    }
    m = atol(argv[2]);
    n = atol(argv[3]);
    if (m < 2 || n < 1)
    {
      std::cerr << "Input time series must have length at least 2 and "
        "count at least 1.\n";
      return 1;
    }
    fill_random(data_vec, m * n);
    comparison(data_vec, m, n);
    return 0;
  }

  std::ifstream input_file(filename);

  if (!input_file.is_open())
  {
    std::cerr << "Unable to open file " << argv[1] << "\n";
    return 1;
  }

  std::string str_buf;
  std::stringstream ss;
  float float_buf;

  while (!input_file.eof())
  {
    getline(input_file, str_buf);
    ss.str(str_buf);
    // first element per line is a class label not a data point.
    bool is_data = false;
    while (!ss.eof())
    {
      ss >> float_buf;
      if (is_data)
      {
        data_vec.push_back(float_buf);
      }
      is_data = true;
    }
    ss.clear();
    n++;
  }
  n--;
  m = data_vec.size() / n;
  // n will overcount by 1 line when we reach the end.
  std::cerr << "Data file " << argv[1] << " contains " << n
    << " time series of length " << m << "\n";

  // Get a pointer to the array data which is dimension (m x n)

  // Let's start checking the performance
  comparison(data_vec, m, n);

  return 0;
}
