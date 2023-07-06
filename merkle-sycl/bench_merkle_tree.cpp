#include <random>
#include <chrono>
#include "bench_merkle_tree.hpp"
#include "merkle_tree.hpp"

uint64_t
benchmark_merklize_approach_1(sycl::queue& q,
                              const size_t leaf_count,
                              const size_t wg_size)
{
  const size_t leaves_size = sizeof(unsigned long) * leaf_count * DIGEST_SIZE;
  const size_t mds_size = sizeof(sycl::ulong4) * STATE_WIDTH * 3;
  const size_t ark_size = sizeof(sycl::ulong4) * NUM_ROUNDS * 3;

  unsigned long* leaves_h = static_cast<unsigned long*>(
    sycl::malloc_host(leaves_size, q));
  unsigned long* leaves_d = static_cast<unsigned long*>(
    sycl::malloc_device(leaves_size, q));

  unsigned long* intermediates_d = static_cast<unsigned long*>(
    sycl::malloc_device(leaves_size, q));

  sycl::ulong4* mds_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(mds_size, q));
  sycl::ulong4* mds_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(mds_size, q));

  sycl::ulong4* ark1_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(ark_size, q));
  sycl::ulong4* ark1_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(ark_size, q));

  sycl::ulong4* ark2_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(ark_size, q));
  sycl::ulong4* ark2_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(ark_size, q));

  {
    std::mt19937 gen(19937);
    std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

    for (uint64_t i = 0; i < leaf_count * DIGEST_SIZE; i++) {
      *(leaves_h + i) = static_cast<unsigned long>(dis(gen));
    }
  }

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  q.memcpy(leaves_d, leaves_h, leaves_size);
  q.memcpy(mds_d, mds_h, mds_size);
  q.memcpy(ark1_d, ark1_h, ark_size);
  q.memcpy(ark2_d, ark2_h, ark_size);

  // wait for host to device copies to complete !
  q.wait();

  // this itself does host synchronization
  auto start_time = std::chrono::high_resolution_clock::now();
  merklize_approach_1(
    q, leaves_d, intermediates_d, leaf_count, wg_size, mds_d, ark1_d, ark2_d);
  auto end_time = std::chrono::high_resolution_clock::now();
  uint64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();

#ifdef DEBUG
  unsigned long *intermediates_h = (unsigned long*) malloc (leaves_size);
  q.memcpy(intermediates_h, intermediates_d, leaves_size).wait();
  for (int i = 0; i < leaf_count * DIGEST_SIZE; i++) {
    printf("%d %lu\n", i, intermediates_h[i]);
  }
  free(intermediates_h);
#endif

  sycl::free(leaves_h, q);
  sycl::free(leaves_d, q);
  sycl::free(intermediates_d, q);
  sycl::free(mds_h, q);
  sycl::free(ark1_h, q);
  sycl::free(ark2_h, q);
  sycl::free(mds_d, q);
  sycl::free(ark1_d, q);
  sycl::free(ark2_d, q);

  return ts;
}

