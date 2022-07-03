#include <chrono>
#include <iostream>
#include <cstdlib>
#include "common.h"
#include "kernels.h"

// thread block size
#define BLOCK_SIZE 256

template <class T>
inline void memcpyH2D(queue &q, buffer<T, 1> &d, const T* h) {
  q.submit([&](handler &cgh) {
    auto acc = d.template get_access<sycl_discard_write>(cgh);
    cgh.copy(h, acc);
  }).wait();
}

template <class T>
void test (queue &q, const int repeat, const int numFloats) 
{
  // Initialize host data, with the first half the same as the second
  T *hostMem = (T*) malloc (sizeof(T) * numFloats);

  srand48(123);
  for (int j = 0; j < numFloats/2 ; ++j)
    hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);

  buffer<T, 1> deviceMem (numFloats);

  range<1> gws (numFloats);
  range<1> lws (BLOCK_SIZE);

  // warmup
  for (int i = 0; i < 4; i++) {
    memcpyH2D(q, deviceMem, hostMem);
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Add1<T>(item, d.get_pointer(), repeat, 10.0);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Add2<T>(item, d.get_pointer(), repeat, 10.0);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Add4<T>(item, d.get_pointer(), repeat, 10.0);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Add8<T>(item, d.get_pointer(), repeat, 10.0);
      });
    });
    q.wait();
  }

  memcpyH2D(q, deviceMem, hostMem);
  auto k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class add1<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Add1<T>(item, d.get_pointer(), repeat, 10.0);
    });
  });
  q.wait();
  auto k_end = std::chrono::high_resolution_clock::now(); 
  auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add1): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class add2<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Add2<T>(item, d.get_pointer(), repeat, 10.0);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add2): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class add4<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Add4<T>(item, d.get_pointer(), repeat, 10.0);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add4): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class add8<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Add8<T>(item, d.get_pointer(), repeat, 10.0);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add8): %f (s)\n", (k_time * 1e-9f));

  // warmup
  for (int i = 0; i < 4; i++) {
    memcpyH2D(q, deviceMem, hostMem);
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Mul1<T>(item, d.get_pointer(), repeat, 1.01);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Mul2<T>(item, d.get_pointer(), repeat, 1.01);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Mul4<T>(item, d.get_pointer(), repeat, 1.01);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        Mul8<T>(item, d.get_pointer(), repeat, 1.01);
      });
    });
    q.wait();
  }

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mul1<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Mul1<T>(item, d.get_pointer(), repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul1): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mul2<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Mul2<T>(item, d.get_pointer(), repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul2): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mul4<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Mul4<T>(item, d.get_pointer(), repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul4): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mul8<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      Mul8<T>(item, d.get_pointer(), repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul8): %f (s)\n", (k_time * 1e-9f));

  // warmup
  for (int i = 0; i < 4; i++) {
    memcpyH2D(q, deviceMem, hostMem);
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MAdd1<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MAdd2<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MAdd4<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MAdd8<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
      });
    });
    q.wait();
  }

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class madd1<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MAdd1<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd1): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class madd2<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MAdd2<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd2): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class madd4<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MAdd4<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd4): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class madd8<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MAdd8<T>(item, d.get_pointer(), repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd8): %f (s)\n", (k_time * 1e-9f));

  // warmup
  for (int i = 0; i < 4; i++) {
    memcpyH2D(q, deviceMem, hostMem);
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MulMAdd1<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MulMAdd2<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MulMAdd4<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
      });
    });
    q.submit([&](handler &cgh) {
      auto d = deviceMem.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        MulMAdd8<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
      });
    });
    q.wait();
  }

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mmadd1<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MulMAdd1<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd1): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mmadd2<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MulMAdd2<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd2): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mmadd4<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MulMAdd4<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd4): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(q, deviceMem, hostMem);
  k_start = std::chrono::high_resolution_clock::now(); 
  q.submit([&](handler &cgh) {
    auto d = deviceMem.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class mmadd8<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
      MulMAdd8<T>(item, d.get_pointer(), repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd8): %f (s)\n", (k_time * 1e-9f));
  
  free(hostMem);
}

int main(int argc, char* argv[]) 
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  // the number of loop iterations inside kernels
  const int repeat = atoi(argv[1]);

  // a multiple of BLOCK_SIZE
  const int numFloats = 2*1024*1024;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  printf("=== Single-precision floating-point kernels ===\n");
  test<float>(q, repeat, numFloats);

  // comment out when double-precision is not supported by a device
  printf("=== Double-precision floating-point kernels ===\n");
  test<double>(q, repeat, numFloats);

  return 0;
}
