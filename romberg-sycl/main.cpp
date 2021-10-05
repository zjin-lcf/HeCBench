#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "reference.h"

// limits of integration
#define A 0
#define B 15

// row size is related to accuracy
#define ROW_SIZE 17
#define EPS      1e-7

inline double f(double x)
{
  return sycl::exp(x)*sycl::sin(x);
}

inline unsigned int getFirstSetBitPos(int n)
{
  return sycl::log2((float)(n&-n))+1;
}

void romberg(double a, double b, double *result, double *smem, nd_item<1> &item)
{
  int threadIdx_x = item.get_local_id(0);
  int blockIdx_x = item.get_group(0);
  int gridDim_x = item.get_group_range(0);
  int blockDim_x = item.get_local_range(0);

  double diff = (b-a)/gridDim_x, step;
  int k;
  int max_eval = (1<<(ROW_SIZE-1));
  b = a + (blockIdx_x+1)*diff;
  a += blockIdx_x*diff;

  step = (b-a)/max_eval;

  double local_col[ROW_SIZE];  // specific to the row size
  for(int i = 0; i < ROW_SIZE; i++) local_col[i] = 0.0;
  if(!threadIdx_x)
  {
    k = blockDim_x;
    local_col[0] = f(a) + f(b);
  }
  else
    k = threadIdx_x;

  for(; k < max_eval; k += blockDim_x)
    local_col[ROW_SIZE - getFirstSetBitPos(k)] += 2.0*f(a + step*k);

  for(int i = 0; i < ROW_SIZE; i++)
    smem[ROW_SIZE*threadIdx_x + i] = local_col[i];
  item.barrier(access::fence_space::local_space);

  if(threadIdx_x < ROW_SIZE)
  {
    double sum = 0.0;
    for(int i = threadIdx_x; i < blockDim_x*ROW_SIZE; i+=ROW_SIZE)
      sum += smem[i];
    smem[threadIdx_x] = sum;
  }

  if(!threadIdx_x)
  {
    double *table = local_col;
    table[0] = smem[0];

    for(int k = 1; k < ROW_SIZE; k++)
      table[k] = table[k-1] + smem[k];

    for(int k = 0; k < ROW_SIZE; k++)  
      table[k]*= (b-a)/(1<<(k+1));

    for(int col = 0 ; col < ROW_SIZE-1 ; col++)
      for(int row = ROW_SIZE-1; row > col; row--)
        table[row] = table[row] + (table[row] - table[row-1])/((1<<(2*col+1))-1);

    result[blockIdx_x] = table[ROW_SIZE-1];
  }
}


int main( int argc, char** argv)
{
  const int numBlocks = 128;
  const int numThreadsPerBlock = 64;

  double *h_result = (double*) malloc (sizeof(double) * numBlocks);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<double, 1> d_result (numBlocks);

  double sum;
  range<1> gws (numBlocks * numThreadsPerBlock);
  range<1> lws (numThreadsPerBlock);
  for (int i = 0; i < 100; i++) {
    q.submit([&] (handler &cgh) {
      auto result = d_result.get_access<sycl_discard_write>(cgh);
      accessor<double, 1, sycl_read_write, access::target::local> smem (ROW_SIZE*numThreadsPerBlock, cgh);
      cgh.parallel_for<class k>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        romberg(A, B, result.get_pointer(), smem.get_pointer(), item);
      });
    });
    q.submit([&] (handler &cgh) {
      auto result = d_result.get_access<sycl_read>(cgh);
      cgh.copy(result, h_result);
    }).wait();
    sum = 0.0;
    for(int k = 0; k < numBlocks; k++) sum += h_result[k];
  }

  // verify
  double ref_sum = reference(f, A, B, ROW_SIZE, EPS);
  printf("%s\n", (fabs(sum - ref_sum) > EPS) ? "FAIL" : "PASS");

  free(h_result);
  return 0;
}

