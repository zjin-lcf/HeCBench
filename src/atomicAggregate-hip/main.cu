#include <chrono>
#include <stdio.h>
#include <hip/hip_runtime.h>

// reference
// https://stackoverflow.com/questions/59879285/whats-the-alternative-for-match-any-sync-on-compute-capability-6

// increment the value at ptr by 1 and return the old value
#define warpSize 32

__device__ int atomicAggInc(int* ptr) {
  int mask;
  //mask = __match_any((unsigned long long)ptr);
  for (int i = 0; i < warpSize; i++){
    unsigned long long tptr = __shfl((unsigned long long)ptr, i);
    unsigned my_mask = __ballot((tptr == (unsigned long long)ptr));
    if (i == (threadIdx.x & (warpSize-1))) mask = my_mask;
  }
  int leader = __ffs(mask) - 1;  // select a leader
  int res = 0;
  unsigned lane_id = threadIdx.x % warpSize;
  if (lane_id == leader) {                 // leader does the update
    res = atomicAdd(ptr, __popc(mask));
  }
  res = __shfl(res, leader);    // get leader’s old value
  return res + __popc(mask & ((1 << lane_id) - 1)); //compute old value
}

__global__ void k(int *d, int s) {
  int *ptr = d + threadIdx.x % s;
  atomicAggInc(ptr);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int nBlocks = 65536;
  const int blockSize = 256;

  for (int ds = 32; ds >= 1; ds = ds / 2) {
    int *d_d, *h_d;
    h_d = new int[ds];
    hipMalloc(&d_d, ds*sizeof(int));
    hipMemset(d_d, 0, ds*sizeof(int));

    hipDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      k<<<nBlocks, blockSize>>>(d_d, ds);
    hipDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    printf("Total kernel time (%d locations): %f (s)\n", ds, time.count());

    hipMemcpy(h_d, d_d, ds*sizeof(int), hipMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < ds; i++) {
      if (h_d[i] != blockSize / ds * nBlocks * repeat) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    hipFree(d_d);
    delete [] h_d;
  }
  return 0;
}
