/*
   Shared memory speeds up performance when we need to access data frequently. 
   Here, the 1D stencil kernel adds all its neighboring data within a radius.

   The C model is added to verify the stencil result on a GPU

   Developer: Zheming Jin
*/

#define LENGTH 134217728
#define THREADS_PER_BLOCK 256
#define RADIUS 7
#define BLOCK_SIZE THREADS_PER_BLOCK

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <assert.h>

void stencil_1d(int *in, int *out, sycl::nd_item<3> item_ct1, int *temp) {

    int gindex = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    int lindex = item_ct1.get_local_id(2) + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];

    // At both end of a block, the sliding window moves beyond the block boundary.
    if (item_ct1.get_local_id(2) < RADIUS) {
       temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
       temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // Synchronize (ensure all the threads will be completed before continue)
    item_ct1.barrier();

    // Apply the 1D stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
       result += temp[lindex + offset];

    // Store the result
    out[gindex] = result; 
}



int main(void) {
   int size = LENGTH * sizeof(int);
   int pad_size = (LENGTH + RADIUS) * sizeof(int);

   int *a, *b;
   // Alloc space for host copies of a, b, c and setup input values
   a = (int *)malloc(pad_size); 
   b = (int *)malloc(size);

   for (int i = 0; i < LENGTH+RADIUS; i++) a[i] = i;

   int *d_a, *d_b;
   // Alloc space for device copies of a, b, c
    dpct::dpct_malloc((void **)&d_a, pad_size);
    dpct::dpct_malloc((void **)&d_b, size);

   // Copy inputs to device
    dpct::dpct_memcpy(d_a, a, pad_size, dpct::host_to_device);

   // Launch add() kernel on GPU
    {
        dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
        dpct::buffer_t d_b_buf_ct1 = dpct::get_buffer(d_b);
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                temp_acc_ct1(sycl::range<1>(270 /*BLOCK_SIZE + 2 * RADIUS*/),
                             cgh);
            auto d_a_acc_ct0 =
                d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
            auto d_b_acc_ct1 =
                d_b_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, LENGTH / THREADS_PER_BLOCK) *
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                    sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                [=](sycl::nd_item<3> item_ct1) {
                    stencil_1d((int *)(&d_a_acc_ct0[0]),
                               (int *)(&d_b_acc_ct1[0]), item_ct1,
                               temp_acc_ct1.get_pointer());
                });
        });
    }

   // Copy result back to host
    dpct::dpct_memcpy(b, d_b, size, dpct::device_to_host);

   // verification
   for (int i = 0; i < 2*RADIUS; i++) {
	   int s = 0;
	   for (int j = i; j <= i+2*RADIUS; j++) {
		   s += j < RADIUS ? 0 : (a[j] - RADIUS);
	   }
	   if (s != b[i]) {
	   	printf("FAILED at %d: %d (cpu) != %d (gpu)\n", i, s, b[i]);
		return 1;
	   }
   }

   for (int i = 2*RADIUS; i < LENGTH; i++) {
	   int s = 0;
	   for (int j = i-RADIUS; j <= i+RADIUS; j++) {
		   s += a[j];
	   }
	   if (s != b[i]) {
	   	printf("FAILED at %d: %d (cpu) != %d (gpu)\n", i, s, b[i]);
		return 1;
	   }
   }

   // Cleanup
   free(a);
   free(b);
    dpct::dpct_free(d_a);
    dpct::dpct_free(d_b);

   printf("PASSED\n");
   return 0;
}
