#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>


void cast1_intrinsics(const int n,
                      const double* input,
                            long long int* output,
                            const sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i >= n) return;

  int r1 = 0;
  unsigned int r2 = 0;
  long long int r3 = 0;
  unsigned long long int r4 = 0;
  
  double x = input[i];

  // __double2hiint call 
  sycl::vec<double, 1> v0{x}; 
  auto v1 = v0.as<sycl::int2>();
  r1 ^= v1[1];

  // __double2loint call 
  r1 ^= v1[0];

  r1 ^= sycl::vec<double, 1>{x}.convert<int, sycl::rounding_mode::rtn>()[0];
  r1 ^= sycl::vec<double, 1>{x}.convert<int, sycl::rounding_mode::rte>()[0];
  r1 ^= sycl::vec<double, 1>{x}.convert<int, sycl::rounding_mode::rtp>()[0];
  r1 ^= sycl::vec<double, 1>{x}.convert<int, sycl::rounding_mode::rtz>()[0];

  r1 ^= sycl::vec<float, 1>{x}.convert<int, sycl::rounding_mode::rtn>()[0];
  r1 ^= sycl::vec<float, 1>{x}.convert<int, sycl::rounding_mode::rte>()[0];
  r1 ^= sycl::vec<float, 1>{x}.convert<int, sycl::rounding_mode::rtp>()[0];
  r1 ^= sycl::vec<float, 1>{x}.convert<int, sycl::rounding_mode::rtz>()[0];

  r1 ^= sycl::bit_cast<int, float>(x);

  r2 ^= sycl::vec<double, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rtn>()[0];
  r2 ^= sycl::vec<double, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rte>()[0];
  r2 ^= sycl::vec<double, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rtp>()[0];
  r2 ^= sycl::vec<double, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rtz>()[0];

  r2 ^= sycl::vec<float, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rtn>()[0];
  r2 ^= sycl::vec<float, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rte>()[0];
  r2 ^= sycl::vec<float, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rtp>()[0];
  r2 ^= sycl::vec<float, 1>{x}
            .convert<unsigned int, sycl::rounding_mode::rtz>()[0];

  r2 ^= sycl::bit_cast<unsigned int, float>(x);

  r3 ^=
      sycl::vec<double, 1>{x}.convert<long long, sycl::rounding_mode::rtn>()[0];
  r3 ^=
      sycl::vec<double, 1>{x}.convert<long long, sycl::rounding_mode::rte>()[0];
  r3 ^=
      sycl::vec<double, 1>{x}.convert<long long, sycl::rounding_mode::rtp>()[0];
  r3 ^=
      sycl::vec<double, 1>{x}.convert<long long, sycl::rounding_mode::rtz>()[0];

  r3 ^=
      sycl::vec<float, 1>{x}.convert<long long, sycl::rounding_mode::rtn>()[0];
  r3 ^=
      sycl::vec<float, 1>{x}.convert<long long, sycl::rounding_mode::rte>()[0];
  r3 ^=
      sycl::vec<float, 1>{x}.convert<long long, sycl::rounding_mode::rtp>()[0];
  r3 ^=
      sycl::vec<float, 1>{x}.convert<long long, sycl::rounding_mode::rtz>()[0];

  r3 ^= sycl::bit_cast<long long>(x);

  r4 ^= sycl::vec<double, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rtn>()[0];
  r4 ^= sycl::vec<double, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rte>()[0];
  r4 ^= sycl::vec<double, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rtp>()[0];
  r4 ^= sycl::vec<double, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rtz>()[0];

  r4 ^= sycl::vec<float, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rtn>()[0];
  r4 ^= sycl::vec<float, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rte>()[0];
  r4 ^= sycl::vec<float, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rtp>()[0];
  r4 ^= sycl::vec<float, 1>{x}
            .convert<unsigned long long, sycl::rounding_mode::rtz>()[0];

  output[i] = (r1 + r2) + (r3 + r4);
}

void cast2_intrinsics(const int n,
                      const long long int* input,
                            long long int* output,
                            const sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i >= n) return;

  float r1 = 0;
  double r2 = 0;
  
  long long int x = input[i];

  int lo = x;
  int hi = x >> 32;
  sycl::int2 v0{lo, hi};
  r1 += v0.as<sycl::vec<double, 1>>();

  r1 += sycl::vec<int, 1>{x}.convert<float, sycl::rounding_mode::rtn>()[0];
  r1 += sycl::vec<int, 1>{x}.convert<float, sycl::rounding_mode::rte>()[0];
  r1 += sycl::vec<int, 1>{x}.convert<float, sycl::rounding_mode::rtp>()[0];
  r1 += sycl::vec<int, 1>{x}.convert<float, sycl::rounding_mode::rtz>()[0];

  r1 += sycl::vec<unsigned int, 1>{x}
            .convert<float, sycl::rounding_mode::rtn>()[0];
  r1 += sycl::vec<unsigned int, 1>{x}
            .convert<float, sycl::rounding_mode::rte>()[0];
  r1 += sycl::vec<unsigned int, 1>{x}
            .convert<float, sycl::rounding_mode::rtp>()[0];
  r1 += sycl::vec<unsigned int, 1>{x}
            .convert<float, sycl::rounding_mode::rtz>()[0];

  r1 += sycl::bit_cast<float, int>(x);
  r1 += sycl::bit_cast<float, unsigned int>(x);

  r1 +=
      sycl::vec<long long, 1>{x}.convert<float, sycl::rounding_mode::rtn>()[0];
  r1 +=
      sycl::vec<long long, 1>{x}.convert<float, sycl::rounding_mode::rte>()[0];
  r1 +=
      sycl::vec<long long, 1>{x}.convert<float, sycl::rounding_mode::rtp>()[0];
  r1 +=
      sycl::vec<long long, 1>{x}.convert<float, sycl::rounding_mode::rtz>()[0];

  r1 += sycl::vec<unsigned long long, 1>{x}
            .convert<float, sycl::rounding_mode::rtn>()[0];
  r1 += sycl::vec<unsigned long long, 1>{x}
            .convert<float, sycl::rounding_mode::rte>()[0];
  r1 += sycl::vec<unsigned long long, 1>{x}
            .convert<float, sycl::rounding_mode::rtp>()[0];
  r1 += sycl::vec<unsigned long long, 1>{x}
            .convert<float, sycl::rounding_mode::rtz>()[0];

  r2 += sycl::vec<int, 1>{x}.convert<double, sycl::rounding_mode::rte>()[0];
  r2 += sycl::vec<unsigned int, 1>{x}
            .convert<double, sycl::rounding_mode::rte>()[0];

  r2 +=
      sycl::vec<long long, 1>{x}.convert<double, sycl::rounding_mode::rtn>()[0];
  r2 +=
      sycl::vec<long long, 1>{x}.convert<double, sycl::rounding_mode::rte>()[0];
  r2 +=
      sycl::vec<long long, 1>{x}.convert<double, sycl::rounding_mode::rtp>()[0];
  r2 +=
      sycl::vec<long long, 1>{x}.convert<double, sycl::rounding_mode::rtz>()[0];

  r2 += sycl::vec<unsigned long long, 1>{x}
            .convert<double, sycl::rounding_mode::rtn>()[0];
  r2 += sycl::vec<unsigned long long, 1>{x}
            .convert<double, sycl::rounding_mode::rte>()[0];
  r2 += sycl::vec<unsigned long long, 1>{x}
            .convert<double, sycl::rounding_mode::rtp>()[0];
  r2 += sycl::vec<unsigned long long, 1>{x}
            .convert<double, sycl::rounding_mode::rtz>()[0];

  r2 += sycl::bit_cast<double>(x);

  output[i] = sycl::bit_cast<long long>(r1 + r2);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t input1_size_bytes = n * sizeof(double);
  const size_t output1_size_bytes = n * sizeof(long long int); 

  const size_t input2_size_bytes = n * sizeof(long long int);
  const size_t output2_size_bytes = n * sizeof(long long int); 

  double *input1 = (double*) malloc (input1_size_bytes);
  long long int *output1 = (long long int*) malloc (output1_size_bytes);

  long long int *input2 = (long long int*) malloc (input2_size_bytes);
  long long int *output2 = (long long int*) malloc (output2_size_bytes);

  for (int i = 1; i <= n; i++) {
    input1[i] = 22.44 / i;
    input2[i] = 0x403670A3D70A3D71;
  }

  double *d_input1;
  long long int *d_output1;
  long long int *d_input2;
  long long int *d_output2;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  d_input1 = (double *)sycl::malloc_device(input1_size_bytes, q);
  q.memcpy(d_input1, input1, input1_size_bytes);

  d_input2 = (long long *)sycl::malloc_device(input2_size_bytes, q);
  q.memcpy(d_input2, input2, input2_size_bytes);

  d_output1 = (long long *)sycl::malloc_device(output1_size_bytes, q);
  d_output2 = (long long *)sycl::malloc_device(output2_size_bytes, q);

  const int grid = (n + 255) / 256;
  const int block = 256;

  sycl::range<1> gws (grid * block);
  sycl::range<1> lws (block);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      cast1_intrinsics(n, d_input1, d_output1, item);
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the cast intrinsics kernel (from FP): %f (us)\n",
         (time * 1e-3f) / repeat);

  q.memcpy(output1, d_output1, output1_size_bytes).wait();

  long long int checksum1 = 0;
  for (int i = 0; i < n; i++) {
    checksum1 = checksum1 ^ output1[i];
  }
  printf("Checksum = %llx\n", checksum1);

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      cast2_intrinsics(n, d_input2, d_output2, item);
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the cast intrinsics kernel (to FP): %f (us)\n",
         (time * 1e-3f) / repeat);

  q.memcpy(output2, d_output2, output2_size_bytes).wait();

  long long int checksum2 = 0;
  for (int i = 0; i < n; i++) {
    checksum2 ^= output2[i];
  }
  printf("Checksum = %llx\n", checksum2);

  sycl::free(d_input1, q);
  sycl::free(d_output1, q);
  sycl::free(d_input2, q);
  sycl::free(d_output2, q);

  free(input1);
  free(output1);
  free(input2);
  free(output2);

  return 0;
}
