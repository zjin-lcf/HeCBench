#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

constexpr int NUM_THREADS      = 128;
constexpr int THREAD_WORK_SIZE = 4;  // elements per thread
constexpr int BLOCK_WORK_SIZE  = THREAD_WORK_SIZE * NUM_THREADS;

void launch_kernel(int64_t N,
                   float* __restrict__ d_o1,
                   float* __restrict__ d_o2,
                   const float* __restrict__ d_x1,
                   const float* __restrict__ d_x2,
                   const float* __restrict__ d_cos,
                   const float* __restrict__ d_sin)
{
  int64_t num_teams = (N + BLOCK_WORK_SIZE - 1) / BLOCK_WORK_SIZE;

  #pragma omp target teams distribute parallel for \
   num_teams(num_teams) num_threads(NUM_THREADS)
  for (int64_t i = 0; i < N; i++) {
    float x1  = d_x1[i];
    float x2  = d_x2[i];
    float c   = d_cos[i];
    float s   = d_sin[i];

    d_o1[i] = x1 * c - x2 * s;
    d_o2[i] = x1 * s + x2 * c;
  }
}

void gpu_kernel_multiple_outputs_impl(int repeat)
{
  int64_t numel = (int64_t)BLOCK_WORK_SIZE * 10000;
  printf("Number of elements: %zu\n", numel);
  printf("Number of teams (blocks): %zu, threads per team: %d\n",
      (size_t)((numel + BLOCK_WORK_SIZE - 1) / BLOCK_WORK_SIZE), NUM_THREADS);

  uint64_t size = numel * sizeof(float);

  float* h_x1  = (float*)malloc(size);
  float* h_x2  = (float*)malloc(size);
  float* h_cos = (float*)malloc(size);
  float* h_sin = (float*)malloc(size);
  float* h_o1  = (float*)malloc(size);
  float* h_o2  = (float*)malloc(size);

  for (int64_t i = 0; i < numel; i++) {
    float fi = i;
    h_x1[i]  = (fi + 1.f) / numel;
    h_x2[i]  = (fi + 1.f) / numel;
    h_cos[i] = cosf(fi / powf(10000.f, fi / numel));
    h_sin[i] = sinf(fi / powf(10000.f, fi / numel));
  }

  #pragma omp target data map(to: h_x1[0:numel], h_x2[0:numel], h_cos[0:numel], h_sin[0:numel]) \
                          map(from: h_o1[0:numel], h_o2[0:numel])
  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      launch_kernel(numel, h_o1, h_o2, h_x1, h_x2, h_cos, h_sin);
    }
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", (time * 1e-3) / repeat);
  }

  // Verify
  bool ok = true;
  for (int64_t i = 0; i < numel; i++) {
    float r1 = h_x1[i] * h_cos[i] - h_x2[i] * h_sin[i];
    float r2 = h_x1[i] * h_sin[i] + h_x2[i] * h_cos[i];
    if (fabsf(r1 - h_o1[i]) > 1e-3f || fabsf(r2 - h_o2[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_x1); free(h_x2); free(h_cos); free(h_sin); free(h_o1); free(h_o2);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  gpu_kernel_multiple_outputs_impl(repeat);
  return 0;
}
