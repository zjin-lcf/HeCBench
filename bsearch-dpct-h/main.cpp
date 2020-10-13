#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

#ifndef Real_t 
#define Real_t float
#endif

#define RPTS 10  // repeat the kernel execution

//#define DEBUG // verify the results of kernel execution


template <typename T>
void
kernel_BS (const T* acc_a, const T* acc_z, size_t* acc_r, const size_t n,
           sycl::nd_item<3> item_ct1) {
  size_t i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
             item_ct1.get_local_id(2);
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
}

template <typename T>
void
kernel_BS2 (const T* acc_a, const T* acc_z, size_t* acc_r, const size_t n,
            sycl::nd_item<3> item_ct1) {
  size_t i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
             item_ct1.get_local_id(2);
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
}

template <typename T>
void
kernel_BS3 (const T* acc_a, const T* acc_z, size_t* acc_r, const size_t n,
            sycl::nd_item<3> item_ct1) {
  size_t i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
             item_ct1.get_local_id(2);
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
}

template <typename T>
void
kernel_BS4 (const T* acc_a, const T* acc_z, size_t* acc_r, const size_t n,
            sycl::nd_item<3> item_ct1, size_t *k) {

  size_t gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
  size_t lid = item_ct1.get_local_id(2);

  if (lid == 0) {
    unsigned nbits = 0;
    while (n >> nbits) nbits++;
    *k = 1ULL << (nbits - 1);
  }
  item_ct1.barrier();

  size_t p = (*k);
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
}

template <typename T>
void bs ( const size_t aSize,
    const size_t zSize,
    const T *a,  // N+1
    const T *z,  // T
    size_t *r,   // T
    const size_t n )
{
  T* buf_x;
  T* buf_z;
  size_t *buf_r;
  dpct::dpct_malloc((void **)&buf_x, sizeof(T) * aSize);
  dpct::dpct_malloc((void **)&buf_z, sizeof(T) * zSize);
  dpct::dpct_malloc((void **)&buf_r, sizeof(size_t) * zSize);
  dpct::dpct_memcpy(buf_x, a, sizeof(T) * aSize, dpct::host_to_device);
  dpct::dpct_memcpy(buf_z, z, sizeof(T) * zSize, dpct::host_to_device);
  {
    std::pair<dpct::buffer_t, size_t> buf_x_buf_ct0 =
        dpct::get_buffer_and_offset(buf_x);
    size_t buf_x_offset_ct0 = buf_x_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> buf_z_buf_ct1 =
        dpct::get_buffer_and_offset(buf_z);
    size_t buf_z_offset_ct1 = buf_z_buf_ct1.second;
    dpct::buffer_t buf_r_buf_ct2 = dpct::get_buffer(buf_r);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto buf_x_acc_ct0 =
          buf_x_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_z_acc_ct1 =
          buf_z_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_r_acc_ct2 =
          buf_r_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, zSize / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                         T *buf_x_ct0 =
                             (T *)(&buf_x_acc_ct0[0] + buf_x_offset_ct0);
                         T *buf_z_ct1 =
                             (T *)(&buf_z_acc_ct1[0] + buf_z_offset_ct1);
                         kernel_BS(buf_x_ct0, buf_z_ct1,
                                   (size_t *)(&buf_r_acc_ct2[0]), n, item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(r, buf_r, sizeof(size_t) * zSize, dpct::device_to_host);
  dpct::dpct_free(buf_x);
  dpct::dpct_free(buf_z);
  dpct::dpct_free(buf_r);
}

template <typename T>
void bs2 ( const size_t aSize,
    const size_t zSize,
    const T *a,  // N+1
    const T *z,  // T
    size_t *r,   // T
    const size_t n )
{
  T* buf_x;
  T* buf_z;
  size_t *buf_r;
  dpct::dpct_malloc((void **)&buf_x, sizeof(T) * aSize);
  dpct::dpct_malloc((void **)&buf_z, sizeof(T) * zSize);
  dpct::dpct_malloc((void **)&buf_r, sizeof(size_t) * zSize);
  dpct::dpct_memcpy(buf_x, a, sizeof(T) * aSize, dpct::host_to_device);
  dpct::dpct_memcpy(buf_z, z, sizeof(T) * zSize, dpct::host_to_device);
  {
    std::pair<dpct::buffer_t, size_t> buf_x_buf_ct0 =
        dpct::get_buffer_and_offset(buf_x);
    size_t buf_x_offset_ct0 = buf_x_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> buf_z_buf_ct1 =
        dpct::get_buffer_and_offset(buf_z);
    size_t buf_z_offset_ct1 = buf_z_buf_ct1.second;
    dpct::buffer_t buf_r_buf_ct2 = dpct::get_buffer(buf_r);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto buf_x_acc_ct0 =
          buf_x_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_z_acc_ct1 =
          buf_z_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_r_acc_ct2 =
          buf_r_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, zSize / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                         T *buf_x_ct0 =
                             (T *)(&buf_x_acc_ct0[0] + buf_x_offset_ct0);
                         T *buf_z_ct1 =
                             (T *)(&buf_z_acc_ct1[0] + buf_z_offset_ct1);
                         kernel_BS2(buf_x_ct0, buf_z_ct1,
                                    (size_t *)(&buf_r_acc_ct2[0]), n, item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(r, buf_r, sizeof(size_t) * zSize, dpct::device_to_host);
  dpct::dpct_free(buf_x);
  dpct::dpct_free(buf_z);
  dpct::dpct_free(buf_r);
}
template <typename T>
void bs3 ( const size_t aSize,
    const size_t zSize,
    const T *a,  // N+1
    const T *z,  // T
    size_t *r,   // T
    const size_t n )
{
  T* buf_x;
  T* buf_z;
  size_t *buf_r;
  dpct::dpct_malloc((void **)&buf_x, sizeof(T) * aSize);
  dpct::dpct_malloc((void **)&buf_z, sizeof(T) * zSize);
  dpct::dpct_malloc((void **)&buf_r, sizeof(size_t) * zSize);
  dpct::dpct_memcpy(buf_x, a, sizeof(T) * aSize, dpct::host_to_device);
  dpct::dpct_memcpy(buf_z, z, sizeof(T) * zSize, dpct::host_to_device);
  {
    std::pair<dpct::buffer_t, size_t> buf_x_buf_ct0 =
        dpct::get_buffer_and_offset(buf_x);
    size_t buf_x_offset_ct0 = buf_x_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> buf_z_buf_ct1 =
        dpct::get_buffer_and_offset(buf_z);
    size_t buf_z_offset_ct1 = buf_z_buf_ct1.second;
    dpct::buffer_t buf_r_buf_ct2 = dpct::get_buffer(buf_r);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto buf_x_acc_ct0 =
          buf_x_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_z_acc_ct1 =
          buf_z_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_r_acc_ct2 =
          buf_r_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, zSize / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                         T *buf_x_ct0 =
                             (T *)(&buf_x_acc_ct0[0] + buf_x_offset_ct0);
                         T *buf_z_ct1 =
                             (T *)(&buf_z_acc_ct1[0] + buf_z_offset_ct1);
                         kernel_BS3(buf_x_ct0, buf_z_ct1,
                                    (size_t *)(&buf_r_acc_ct2[0]), n, item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(r, buf_r, sizeof(size_t) * zSize, dpct::device_to_host);
  dpct::dpct_free(buf_x);
  dpct::dpct_free(buf_z);
  dpct::dpct_free(buf_r);
}
template <typename T>
void bs4 ( const size_t aSize,
    const size_t zSize,
    const T *a,  // N+1
    const T *z,  // T
    size_t *r,   // T
    const size_t n )
{
  T* buf_x;
  T* buf_z;
  size_t *buf_r;
  dpct::dpct_malloc((void **)&buf_x, sizeof(T) * aSize);
  dpct::dpct_malloc((void **)&buf_z, sizeof(T) * zSize);
  dpct::dpct_malloc((void **)&buf_r, sizeof(size_t) * zSize);
  dpct::dpct_memcpy(buf_x, a, sizeof(T) * aSize, dpct::host_to_device);
  dpct::dpct_memcpy(buf_z, z, sizeof(T) * zSize, dpct::host_to_device);
  {
    std::pair<dpct::buffer_t, size_t> buf_x_buf_ct0 =
        dpct::get_buffer_and_offset(buf_x);
    size_t buf_x_offset_ct0 = buf_x_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> buf_z_buf_ct1 =
        dpct::get_buffer_and_offset(buf_z);
    size_t buf_z_offset_ct1 = buf_z_buf_ct1.second;
    dpct::buffer_t buf_r_buf_ct2 = dpct::get_buffer(buf_r);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<size_t, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          k_acc_ct1(cgh);
      auto buf_x_acc_ct0 =
          buf_x_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_z_acc_ct1 =
          buf_z_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto buf_r_acc_ct2 =
          buf_r_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, zSize / 256) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
          [=](sycl::nd_item<3> item_ct1) {
            T *buf_x_ct0 = (T *)(&buf_x_acc_ct0[0] + buf_x_offset_ct0);
            T *buf_z_ct1 = (T *)(&buf_z_acc_ct1[0] + buf_z_offset_ct1);
            kernel_BS4(buf_x_ct0, buf_z_ct1, (size_t *)(&buf_r_acc_ct2[0]), n,
                       item_ct1, k_acc_ct1.get_pointer());
          });
    });
  }
  dpct::dpct_memcpy(r, buf_r, sizeof(size_t) * zSize, dpct::device_to_host);
  dpct::dpct_free(buf_x);
  dpct::dpct_free(buf_z);
  dpct::dpct_free(buf_r);
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


  for(uint k = 0; k < RPTS; k++) {
    bs(aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs1");
#endif

  for(uint k = 0; k < RPTS; k++) {
    bs2(aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs2");
#endif

  for(uint k = 0; k < RPTS; k++) {
    bs3(aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs3");
#endif

  for(uint k = 0; k < RPTS; k++) {
    bs4(aSize, zSize, a, z, r, N);  
  }
#ifdef DEBUG
  verify(a, z, r, aSize, zSize, "bs4");
#endif


  free(a);
  free(z);
  free(r);
  return 0;
}
