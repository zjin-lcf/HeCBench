/**
 * main-omp.cpp: This file is the modified read-only mixbench GPU micro-benchmark suite.
 *
 **/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1

void benchmark_func(float *g_data, const int blockdim, const int compute_iterations,
                    sycl::nd_item<3> item_ct1) {
	const unsigned int blockSize = blockdim;
	const int stride = blockSize;
 int idx =
     item_ct1.get_group(2) * blockSize * granularity + item_ct1.get_local_id(2);
 const int big_stride = item_ct1.get_group_range(2) * blockSize * granularity;

        float tmps[granularity];
	for(int k=0; k<fusion_degree; k++){
		#pragma unroll
		for(int j=0; j<granularity; j++){
			// Load elements (memory intensive part)
			tmps[j] = g_data[idx+j*stride+k*big_stride];
			// Perform computations (compute intensive part)
			for(int i=0; i<compute_iterations; i++){
				tmps[j] = tmps[j]*tmps[j]+(float)seed;
			}
		}
		// Multiply add reduction
		float sum = 0;
		#pragma unroll
		for(int j=0; j<granularity; j+=2)
			sum += tmps[j]*tmps[j+1];
	  #pragma unroll
		for(int j=0; j<granularity; j++)
			g_data[idx+k*big_stride] = sum;
	}
}

void mixbenchGPU(long size) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        const char *benchtype = "compute with global memory (block strided)";
	printf("Trade-off type:%s\n", benchtype);
	float *cd = (float*) malloc (size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

	const long reduced_grid_size = size/granularity/128;
	const int block_dim = 256;
	const int grid_dim = reduced_grid_size/block_dim;

  float *d_cd;
 d_cd = sycl::malloc_device<float>(size, q_ct1);
 q_ct1.memcpy(d_cd, cd, size * sizeof(float)).wait();

  for (int compute_iterations = 0; compute_iterations < 2048; compute_iterations++) {
  q_ct1.submit([&](sycl::handler &cgh) {
   cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_dim) *
                                          sycl::range<3>(1, 1, block_dim),
                                      sycl::range<3>(1, 1, block_dim)),
                    [=](sycl::nd_item<3> item_ct1) {
                     benchmark_func(d_cd, block_dim, compute_iterations,
                                    item_ct1);
                    });
  });
  }
 q_ct1.memcpy(cd, d_cd, size * sizeof(float)).wait();

  // verification
  for (int i = 0; i < size; i++) 
    if (cd[i] != 0) {
   if (fabsf(cd[i] - 0.050807f) > 1e-6f)
        printf("Verification failed at index %d: %f\n", i, cd[i]);
    }

  free(cd);
 sycl::free(d_cd, q_ct1);
}


int main(int argc, char* argv[]) {

	unsigned int datasize = VECTOR_SIZE*sizeof(float);

	printf("Buffer size: %dMB\n", datasize/(1024*1024));
	
	mixbenchGPU(VECTOR_SIZE);

	return 0;
}
