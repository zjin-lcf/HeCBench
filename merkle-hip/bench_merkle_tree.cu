#include <random>
#include <chrono>
#include "bench_merkle_tree.hpp"
#include "merkle_tree.hpp"

uint64_t
benchmark_merklize_approach_1(const size_t leaf_count,
                              const size_t wg_size)
{
  const size_t leaves_size = sizeof(ulong) * leaf_count * DIGEST_SIZE;
  const size_t mds_size = sizeof(ulong4) * STATE_WIDTH * 3;
  const size_t ark_size = sizeof(ulong4) * NUM_ROUNDS * 3;

  ulong* leaves_h;
  hipHostMalloc((void**)&leaves_h, leaves_size);

  ulong* leaves_d;
  hipMalloc((void**)&leaves_d, leaves_size);

  ulong* intermediates_d;
  hipMalloc((void**)&intermediates_d, leaves_size);

  ulong4* mds_h;
  hipHostMalloc((void**)&mds_h, mds_size);

  ulong4* mds_d;
  hipMalloc((void**)&mds_d, mds_size);

  ulong4* ark1_h;
  hipHostMalloc((void**)&ark1_h, ark_size);

  ulong4* ark1_d;
  hipMalloc((void**)&ark1_d, ark_size);

  ulong4* ark2_h;
  hipHostMalloc((void**)&ark2_h, ark_size);

  ulong4* ark2_d;
  hipMalloc((void**)&ark2_d, ark_size);

  {
    std::mt19937 gen(19937);
    std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

    for (uint64_t i = 0; i < leaf_count * DIGEST_SIZE; i++) {
      *(leaves_h + i) = static_cast<ulong>(dis(gen));
    }
  }

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  hipMemcpy(leaves_d, leaves_h, leaves_size, hipMemcpyHostToDevice);
  hipMemcpy(mds_d, mds_h, mds_size, hipMemcpyHostToDevice);
  hipMemcpy(ark1_d, ark1_h, ark_size, hipMemcpyHostToDevice);
  hipMemcpy(ark2_d, ark2_h, ark_size, hipMemcpyHostToDevice);

  // this itself does host synchronization
  auto start_time = std::chrono::high_resolution_clock::now();
  merklize_approach_1(
    leaves_d, intermediates_d, leaf_count, wg_size, mds_d, ark1_d, ark2_d);
  auto end_time = std::chrono::high_resolution_clock::now();
  uint64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();

#ifdef DEBUG
  ulong *intermediates_h = (ulong*) malloc (leaves_size);
  hipMemcpy(intermediates_h, intermediates_d, leaves_size, hipMemcpyDeviceToHost);
  for (int i = 0; i < leaf_count * DIGEST_SIZE; i++) {
    printf("%d %lu\n", i, intermediates_h[i]);
  }
  free(intermediates_h);
#endif

  hipHostFree(leaves_h);
  hipHostFree(mds_h);
  hipHostFree(ark1_h);
  hipHostFree(ark2_h);

  hipFree(leaves_d);
  hipFree(intermediates_d);
  hipFree(mds_d);
  hipFree(ark1_d);
  hipFree(ark2_d);

  return ts;
}
