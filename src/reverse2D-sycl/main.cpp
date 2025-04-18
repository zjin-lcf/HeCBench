#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "reverse.hpp"

template <typename T>
void eval_case(sycl::queue &q, T* d_in, T* h_in, T* d_out, T* h_out, 
               bool rowMajor, bool alongRows,
               int nrows, int ncols, const int repeat)
{
  // Debug the reverse of a 8x8 matrix for more details
  printf("\nInput matrix is %s major and reverse along %s\n",
         rowMajor ? "row" : "column", alongRows ? "rows" : "columns" );

  const size_t matrix_size = (size_t)nrows * ncols;
  const size_t elem_size = sizeof(T) * matrix_size;
  q.memcpy(d_in, h_in, elem_size).wait();

  long time = 0;
  for (int i = 0; i < repeat; i++)
    time += reverse(d_out, d_in, nrows, ncols, rowMajor, alongRows, q);
  printf("Average kernel execution time: %f (ms)\n", time * 1e-6f / repeat);

#ifdef DEBUG
  q.memcpy(h_out, d_out, elem_size).wait();
  for (size_t i = 1; i <= matrix_size; i++) {
    printf("%d ", h_out[i-1]);
    if (i % ncols == 0) printf("\n");
  }
#endif
}

template <typename T>
void eval(sycl::queue &q, const int nrows, const int ncols, const int repeat)
{
  const size_t matrix_size = (size_t)nrows * ncols;
  const size_t elem_size = sizeof(T) * matrix_size;

  T *h_in = (T*) malloc (elem_size);
  T *h_out = (T*) malloc (elem_size);
  T *d_in = (T *)sycl::malloc_device(elem_size, q);
  T *d_out = (T *)sycl::malloc_device(elem_size, q);

  std::default_random_engine generator (123);
  std::uniform_int_distribution<int> distribution(0, 255);
 
#ifdef DEBUG
  printf("Input matrix:\n");
#endif
  for (size_t i = 1; i <= matrix_size; i++) {
    h_in[i-1] = static_cast<T>(distribution(generator));
#ifdef DEBUG
    printf("%d ", h_in[i-1]);
    if (i % ncols == 0) printf("\n");
#endif
  }
 
  // specify the number of test cases
  eval_case(q, d_in, h_in, d_out, h_out, true, true, nrows, ncols, repeat); 
  eval_case(q, d_in, h_in, d_out, h_out, true, false, nrows, ncols, repeat); 
  eval_case(q, d_in, h_in, d_out, h_out, false, true, nrows, ncols, repeat); 
  eval_case(q, d_in, h_in, d_out, h_out, false, false, nrows, ncols, repeat); 
  
  free(h_in);
  free(h_out);
  sycl::free(d_in, q);
  sycl::free(d_out, q);
}

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <nrows> <ncols> <iterations>\n", argv[0]);
    return 1;
  }

  const int nrows = atoi(argv[1]);
  const int ncols = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("\nThe size of each matrix element is %zu byte\n", sizeof(unsigned char));
  eval<unsigned char>(q, nrows, ncols, repeat);
  printf("\nThe size of each matrix element is %zu bytes\n", sizeof(ushort));
  eval<ushort>(q, nrows, ncols, repeat);
  printf("\nThe size of each matrix element is %zu bytes\n", sizeof(uint));
  eval<uint>(q, nrows, ncols, repeat);
  printf("\nThe size of each matrix element is %zu bytes\n", sizeof(ulong));
  eval<ulong>(q, nrows, ncols, repeat);

  return 0;
}
