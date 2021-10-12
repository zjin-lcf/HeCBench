#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include "common.h"

#define ERT_ALIGN           256
#define ERT_NUM_EXPERIMENTS 1
//#define ERT_MEMORY_MAX      33554432
#define ERT_MEMORY_MAX      1000
#define ERT_WORKING_SET_MIN 128
#define ERT_TRIALS_MIN      1
#define ERT_WSS_MULT        1.3

#include "kernel.h"

double getTime()
{
  double time;
  struct timeval tm;
  gettimeofday(&tm, NULL);
  time = tm.tv_sec + (tm.tv_usec / 1000000.0);
  return time;
}

template <typename T>
T *alloc(uint64_t psize)
{
  T* buffer =  (T *)malloc(psize);
  if (buffer == nullptr) {
    fprintf(stderr, "Out of memory!\n");
    exit(1);
  }
  return buffer;
}

template <typename T>
inline void launchKernel(queue &q, uint64_t n, uint64_t t, T *buf, buffer<T,1> &d_buf,
                         int *bytes_per_elem_ptr, int *mem_accesses_per_elem_ptr)
{
  gpuKernel<T>(q, n, t, d_buf, bytes_per_elem_ptr, mem_accesses_per_elem_ptr);
}

template <typename T>
void run(queue &q, uint64_t PSIZE, T *buf)
{

  uint64_t nsize = PSIZE;
  nsize          = nsize & (~(ERT_ALIGN - 1));
  nsize          = nsize / sizeof(T);

  buffer<T, 1> d_buf (nsize);

  q.submit([&] (handler &cgh) {
    auto acc = d_buf.template get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, (T)0);
  });

  uint64_t n, nNew;
  uint64_t t;
  int bytes_per_elem;
  int mem_accesses_per_elem;

  n = ERT_WORKING_SET_MIN;
  while (n <= nsize) { // working set - nsize

    uint64_t ntrials = nsize / n;
    if (ntrials < ERT_TRIALS_MIN)
      ntrials = ERT_TRIALS_MIN;

    // initialize small chunck of buffer within each thread
    float value = -1.f;
    initialize<T>(nsize, buf, value);

    q.submit([&] (handler &cgh) {
      auto acc = d_buf.template get_access<sycl_discard_write>(cgh);
      cgh.copy(buf, acc);
    });

    for (t = 1; t <= ntrials; t = t * 2) { // working set - ntrials
      launchKernel<T>(q, n, t, buf, d_buf, &bytes_per_elem, &mem_accesses_per_elem);
    } // working set - ntrials

    q.submit([&] (handler &cgh) {
      auto acc = d_buf.template get_access<sycl_read>(cgh);
      cgh.copy(acc, buf);
    }).wait();

    nNew = ERT_WSS_MULT * n;
    if (nNew == n) {
      nNew = n + 1;
    }

    n = nNew;
  } // working set - nsize
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "Usage: %s gpu_blocks gpu_threads\n", argv[0]);
    return -1;
  }

  gpu_blocks  = atoi(argv[1]);
  gpu_threads = atoi(argv[2]);
  printf("\n");
  printf("GPU_BLOCKS     %d\n", gpu_blocks);
  printf("GPU_THREADS    %d\n", gpu_threads);

  uint64_t TSIZE = ERT_MEMORY_MAX;
  uint64_t PSIZE = TSIZE;
  double start, checksum;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // FP16
  half2 *hlfbuf = alloc<half2>(PSIZE);
  start = getTime();
  run<half2>(q, PSIZE, hlfbuf);
  printf("runtime (half2): %lf (s)\n", getTime() - start);
  // final checksum
  checksum = 0; 
  for (uint64_t i = 0; i < PSIZE / sizeof(half2); i++) {
    float2 t = hlfbuf[i].convert<float, sycl::rounding_mode::automatic>();
    checksum += t.x() + t.y();
  }
  printf("checksum: %lf\n", checksum);
  free(hlfbuf);

  // FP32
  float *sglbuf = alloc<float>(PSIZE);
  start = getTime();
  run<float>(q, PSIZE, sglbuf);
  printf("runtime (float): %lf (s)\n", getTime() - start);
  // final checkchecksum
  checksum = 0; 
  for (uint64_t i = 0; i < PSIZE/sizeof(float); i++) checksum += sglbuf[i];
  printf("checksum: %lf\n", checksum);
  free(sglbuf);

  // FP64
  double *dblbuf = alloc<double>(PSIZE);
  start = getTime();
  run<double>(q, PSIZE, dblbuf);
  printf("runtime (double): %lf (s)\n", getTime() - start);
  // final checkchecksum
  checksum = 0; 
  for (uint64_t i = 0; i < PSIZE/sizeof(double); i++) checksum += dblbuf[i];
  printf("checksum: %lf\n", checksum);
  free(dblbuf);

  return 0;
}
