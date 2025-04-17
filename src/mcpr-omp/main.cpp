#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include "kernels.h"
#include "reference.h"

// transpose
double* t(const double *idata, const int width, const int height)
{
  double *odata = (double*) malloc (sizeof(double) * width * height); 
  for (int yIndex = 0; yIndex < height; yIndex++) {
    for (int xIndex = 0; xIndex < width; xIndex++) {
      int index_in  = xIndex + width * yIndex;
      int index_out = yIndex + height * xIndex;
      odata[index_out] = idata[index_in];
    }
  }
  return odata;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <path to filename> <repeat>\n", argv[0]);
    return 1;
  }
  char *filename = argv[1];
  const int repeat = atoi(argv[2]);

  // n and K should match the dimension of the dataset in the csv file
  const int n = 26280, K = 21, M = 10000;

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: failed to open file alphas.csv. Exit\n");
    return 1;
  }

  int alphas_size = n * K; // n rows and K cols
  int alphas_size_byte = n * K * sizeof(double);

  int rands_size = M * K;  // M rows and K cols
  int rands_size_byte = M * K * sizeof(double);

  double *alphas, *rands, *probs, *probs_ref;
  alphas = (double*) malloc (alphas_size_byte);
  rands = (double*) malloc (rands_size_byte);
  probs = (double*) malloc (alphas_size_byte);
  probs_ref = (double*) malloc (alphas_size_byte);

  // load the csv file 
  for (int i = 0; i < alphas_size; i++)
    fscanf(fp, "%lf", &alphas[i]);
  fclose(fp);

  // normal distribution (mean: 0 and var: 1)
  std::mt19937 gen(19937);
  std::normal_distribution<double> norm_dist(0.0,1.0);
  for (int i = 0; i < rands_size; i++) rands[i] = norm_dist(gen); 

  reference(alphas, rands, probs_ref, n, K, M);

  #pragma omp target data map (to: rands[0:rands_size], alphas[0:alphas_size]) \
                          map (alloc: probs[0:alphas_size])
  {
    // kernel 1
    int threads_per_block = 192;
    int num_blocks = ceil(1.0 * n / threads_per_block);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      compute_probs(num_blocks, threads_per_block, alphas, rands, probs, n, K, M);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of compute_probs kernel: %f (s)\n", (time * 1e-9f) / repeat);

    #pragma omp target update from (probs[0:alphas_size])

    verify(probs, probs_ref, alphas_size);

    // kernel 2
    double *t_rands = t(rands, K, M);
    double *t_alphas = t(alphas, K, n);

    reference_unitStrides(t_alphas, t_rands, probs_ref, n, K, M);

    memcpy(rands, t_rands, rands_size_byte);
    memcpy(alphas, t_alphas, alphas_size_byte);

    #pragma omp target update to (rands[0:rands_size])
    #pragma omp target update to (alphas[0:alphas_size])

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      compute_probs_unitStrides(num_blocks, threads_per_block,
                                alphas, rands, probs, n, K, M);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    #pragma omp target update from (probs[0:alphas_size])

    verify(probs, probs_ref, alphas_size);

    // kernel 3
    threads_per_block = 96;
    num_blocks = ceil(1.0 * n / threads_per_block);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      compute_probs_unitStrides_sharedMem(num_blocks, threads_per_block,
                                          alphas, rands, probs, n, K, M);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    #pragma omp target update from (probs[0:alphas_size])

    verify(probs, probs_ref, alphas_size);
 
    free(t_alphas);
    free(t_rands);
  }

  // free memory 
  free(alphas);
  free(rands);
  free(probs);
  free(probs_ref);
  return 0;
}
