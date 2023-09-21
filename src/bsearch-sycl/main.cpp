#include <cstdlib>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include "bs.h"
#include "bs2.h"
#include "bs3.h"
#include "bs4.h"

#ifndef Real_t 
#define Real_t float
#endif

#ifdef DEBUG
void verify(Real_t *a, Real_t *z, size_t *r, size_t aSize, size_t zSize, std::string msg)
{
    for (size_t i = 0; i < zSize; ++i)
    {
        // check result
        if (!(r[i]+1 < aSize && a[r[i]] <= z[i] && z[i] < a[r[i] + 1]))
        {
          std::cout << msg << ": incorrect result:" << std::endl;
          std::cout << "index = " << i << " r[index] = " << r[i] << std::endl;
          std::cout << a[r[i]] << " <= " << z[i] << " < " << a[r[i] + 1] << std::endl;
          break;
        }
        // clear result
        r[i] = 0xFFFFFFFF;
    }
}
#endif

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Usage ./main <number of elements> <repeat>\n";
    return 1;
  }

  size_t numElem = atol(argv[1]);
  uint repeat = atoi(argv[2]);

  srand(2);
  size_t aSize = numElem;
  size_t zSize = 2*aSize;
  Real_t *a = NULL;
  Real_t *z = NULL;
  size_t *r = NULL;
  posix_memalign((void**)&a, 1024, aSize * sizeof(Real_t));
  posix_memalign((void**)&z, 1024, zSize * sizeof(Real_t));
  posix_memalign((void**)&r, 1024, zSize * sizeof(size_t));

  size_t N = aSize-1;

  // strictly ascending
  for (size_t i = 0; i < aSize; i++) a[i] = i;

  // lower = 0, upper = n-1
  for (size_t i = 0; i < zSize; i++) z[i] = rand() % N;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Real_t *d_a = sycl::malloc_device<Real_t>(aSize, q);
  q.memcpy(d_a, a, aSize * sizeof(Real_t));

  Real_t *d_z = sycl::malloc_device<Real_t>(zSize, q);
  q.memcpy(d_z, z, zSize * sizeof(Real_t));

  size_t *d_r = sycl::malloc_device<size_t>(zSize, q);

  bs(q, aSize, zSize, d_a, d_z, d_r, N, repeat);  

#ifdef DEBUG
  q.memcpy(r, d_r, zSize * sizeof(size_t)).wait();
  verify(a, z, r, aSize, zSize, "bs");
#endif

  bs2(q, aSize, zSize, d_a, d_z, d_r, N, repeat);  

#ifdef DEBUG
  q.memcpy(r, d_r, zSize * sizeof(size_t)).wait();
  verify(a, z, r, aSize, zSize, "bs2");
#endif

  bs3(q, aSize, zSize, d_a, d_z, d_r, N, repeat);  

#ifdef DEBUG
  q.memcpy(r, d_r, zSize * sizeof(size_t)).wait();
  verify(a, z, r, aSize, zSize, "bs3");
#endif

  bs4(q, aSize, zSize, d_a, d_z, d_r, N, repeat);  

#ifdef DEBUG
  q.memcpy(r, d_r, zSize * sizeof(size_t)).wait();
  verify(a, z, r, aSize, zSize, "bs4");
#endif

  sycl::free(d_a, q);
  sycl::free(d_z, q);
  sycl::free(d_r, q);
  free(a);
  free(z);
  free(r);
  return 0;
}
