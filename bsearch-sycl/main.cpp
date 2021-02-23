#include <iostream>
#include "common.h"


#ifndef Real_t 
#define Real_t float
#endif

#define RPTS 10  // repeat the kernel execution

//#define DEBUG // verify the results of kernel execution

template <typename T>
class BS;

template <typename T>
void bs ( queue &q,
          const size_t aSize,
          const size_t zSize,
          const T *a,  // N+1
          const T *z,  // T
               size_t *r,  // T
          const size_t n )
{
  buffer<T, 1> buf_x(a, aSize);
  buffer<T, 1> buf_z(z, zSize);
  buffer<size_t, 1> buf_r(r, zSize);
  nd_range<1> ndr{range<1>(zSize), range<1>(256)};

  q.submit([&](handler& cgh) {
    auto acc_a = buf_x.template get_access<sycl_read>(cgh);
    auto acc_z = buf_z.template get_access<sycl_read>(cgh);
    auto acc_r = buf_r.template get_access<sycl_discard_write>(cgh);

    cgh.parallel_for<class BS<T>>(ndr, [=](nd_item<1> item) {
      size_t i = item.get_global_id(0);
       T z = acc_z[i];
       size_t low = 0;
       size_t high = n;
        while (high - low > 1) {
          size_t mid = low + (high - low)/2;
          if (z < acc_a[mid])
            high = mid;
          else
            low = mid;
        }
        acc_r[i] = low;
    });
  });
  q.wait();
}

template <typename T>
class BS2;

template <typename T>
void bs2 (queue &q,
          const size_t aSize,
          const size_t zSize,
          const T *a,  // N+1
          const T *z,  // T
          size_t *r,  // T
          const size_t n )
{
  buffer<T, 1> buf_x(a, aSize);
  buffer<T, 1> buf_z(z, zSize);
  buffer<size_t, 1> buf_r(r, zSize);
  nd_range<1> ndr{range<1>(zSize), range<1>(256)};

  q.submit([&](handler& cgh) {
    auto acc_a = buf_x.template get_access<sycl_read>(cgh);
    auto acc_z = buf_z.template get_access<sycl_read>(cgh);
    auto acc_r = buf_r.template get_access<sycl_discard_write>(cgh);

    cgh.parallel_for<class BS2<T>>(ndr, [=](nd_item<1> item) {
      size_t i = item.get_global_id(0);
      unsigned  nbits = 0;
      while (n >> nbits) nbits++;
      size_t k = 1ULL << (nbits - 1);
      T z = acc_z[i];
      size_t idx = (acc_a[k] <= z) ? k : 0;
      while (k >>= 1) {
        size_t r = idx | k;
        if (r < n && z >= acc_a[r]) { 
          idx = r;
        }
      }
      acc_r[i] = idx;
    });
  });
  q.wait();
}

template <typename T>
class BS3;

template <typename T>
void bs3 ( queue &q,
          const size_t aSize,
          const size_t zSize,
          const T *a,  // N+1
          const T *z,  // T
          size_t *r,  // T
          const size_t n )
{
  buffer<T, 1> buf_x(a, aSize);
  buffer<T, 1> buf_z(z, zSize);
  buffer<size_t, 1> buf_r(r, zSize);
  nd_range<1> ndr{range<1>(zSize), range<1>(256)};

  q.submit([&](handler& cgh) {
    auto acc_a = buf_x.template get_access<sycl_read>(cgh);
    auto acc_z = buf_z.template get_access<sycl_read>(cgh);
    auto acc_r = buf_r.template get_access<sycl_discard_write>(cgh);

    cgh.parallel_for<class BS3<T>>(ndr, [=] (nd_item<1> item) {
      size_t i = item.get_global_id(0);
       unsigned nbits = 0;
       while (n >> nbits) nbits++;
       size_t k = 1ULL << (nbits - 1);
       T z = acc_z[i];
       size_t idx = (acc_a[k] <= z) ? k : 0;
       while (k >>= 1) {
         size_t r = idx | k;
         size_t w = r < n ? r : n; 
         if (z >= acc_a[w]) { 
           idx = r;
         }
       }
       acc_r[i] = idx;
    });
  });
  q.wait();
}

template <typename T>
class BS4;

template <typename T>
void bs4 (queue &q,
          const size_t aSize,
          const size_t zSize,
          const T *a,  // N+1
          const T *z,  // T
          size_t *r,  // T
          const size_t n
    )
{

  buffer<T, 1> buf_x(a, aSize);
  buffer<T, 1> buf_z(z, zSize);
  buffer<size_t, 1> buf_r(r, zSize);
  nd_range<1> ndr{range<1>(zSize), range<1>(256)};

  q.submit([&](handler& cgh) {
    auto acc_a = buf_x.template get_access<sycl_read>(cgh);
    auto acc_z = buf_z.template get_access<sycl_read>(cgh);
    auto acc_r = buf_r.template get_access<sycl_discard_write>(cgh);

    accessor<size_t, 1, sycl_read_write, access::target::local> k(range<1>(1), cgh);

    cgh.parallel_for<class BS4<T>>(ndr, [=](nd_item<1> item) {
       size_t gid = item.get_global_id(0);
       size_t lid = item.get_local_id(0);

       if (lid == 0) {
         unsigned nbits = 0;
         while (n >> nbits) nbits++;
         k[0] = 1ULL << (nbits - 1);
       }
       item.barrier(access::fence_space::local_space);

       size_t p = k[0];
       T z = acc_z[gid];
       size_t idx = (acc_a[p] <= z) ? p : 0;
       while (p >>= 1) {
         size_t r = idx | p;
         size_t w = r < n ? r : n;
         if (z >= acc_a[w]) { 
           idx = r;
         }
       }
       acc_r[gid] = idx;
    });
  });
  q.wait();
}

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

int main(int argc, char* argv[])
{
  srand(2);
  size_t numElem = atol(argv[1]);
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
  for (size_t i = 0; i < zSize; i++) { 
    z[i] = rand() % N;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);


  // bs1
  for(uint k = 0; k < RPTS; k++) {
    bs(q, aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs1");
#endif

  // bs2
  for(uint k = 0; k < RPTS; k++) {
    bs2(q, aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs2");
#endif

  for(uint k = 0; k < RPTS; k++) {
    bs3(q, aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs3");
#endif

  for(uint k = 0; k < RPTS; k++) {
    bs4(q, aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs4");
#endif


  free(a);
  free(z);
  free(r);
  return 0;
}
