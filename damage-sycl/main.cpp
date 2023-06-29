#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernel.h"

// threads per block
#define BS 256

double LCG_random_double(uint64_t * seed)
{
  const unsigned long m = 9223372036854775808ULL; // 2^63
  const unsigned long a = 2806196910506780709ULL;
  const unsigned long c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }

  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int m = (n + BS - 1) / BS; // number of groups

  int *nlist = (int*) malloc (sizeof(int) * n);
  int *family = (int*) malloc (sizeof(int) * m);
  int *n_neigh = (int*) malloc (sizeof(int) * m);
  double *damage = (double*) malloc (sizeof(double) * m);

  unsigned long seed = 123;
  for (int i = 0; i < n; i++) {
    nlist[i] = (LCG_random_double(&seed) > 0.5) ? 1 : -1;
  }

  for (int i = 0; i < m; i++) {
    int s = 0;
    for (int j = 0; j < BS; j++) {
      s += (nlist[i*BS+j] != -1) ? 1 : 0;
    }
    // non-zero values
    family[i] = s + 1 + s * LCG_random_double(&seed);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_nlist = sycl::malloc_device<int>(n, q);
  q.memcpy(d_nlist, nlist, sizeof(int) * n);

  int *d_family = sycl::malloc_device<int>(m, q);
  q.memcpy(d_family, family, sizeof(int) * m);

  int *d_n_neigh = sycl::malloc_device<int>(m, q);
  q.memcpy(d_n_neigh, n_neigh, sizeof(int) * m);

  double *d_damage = sycl::malloc_device<double>(m, q);

  sycl::range<1> lws (BS);
  sycl::range<1> gws (m*BS);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int, 1> sm (sycl::range<1>(BS), cgh);
      cgh.parallel_for<class compute>(
         sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
         damage_of_node(item,
                        n,
                        d_nlist,
                        d_family,
                        d_n_neigh,
                        d_damage,
                        sm.get_pointer());
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(damage, d_damage, sizeof(double) * m).wait();
  double sum = 0.0;
  for (int i = 0; i < m; i++) sum += damage[i]; 
  printf("Checksum: total damage = %lf\n", sum);

  sycl::free(d_nlist, q);
  sycl::free(d_family, q);
  sycl::free(d_n_neigh, q);
  sycl::free(d_damage, q);
  free(nlist);
  free(family);
  free(n_neigh);
  free(damage);

  return 0;
}
