/*
   Shared memory speeds up performance when we need to access data frequently. 
   Here, the 1D stencil kernel adds all its neighboring data within a radius.

   The C model is added to verify the stencil result on a GPU

   Developer: Zheming Jin
*/

#define LENGTH 134217728
//#define LENGTH 2048
#define THREADS_PER_BLOCK 256
#define RADIUS 7
#define BLOCK_SIZE THREADS_PER_BLOCK

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


int main(void) {
   int size = LENGTH;
   int pad_size = (LENGTH + RADIUS);

   // Alloc space for host copies of a, b, c and setup input values
   int* a = (int *)malloc(pad_size*sizeof(int)); 
   int* b = (int *)malloc(size*sizeof(int));

   for (int i = 0; i < LENGTH+RADIUS; i++) a[i] = i;

#pragma omp target teams distribute map(to: a[0:pad_size]) map(from:b[0:size]) 
   for (int i = 0; i < LENGTH; i = i + THREADS_PER_BLOCK) {
     int temp[BLOCK_SIZE + 2 * RADIUS];
     #pragma omp parallel for schedule(static,1)
     for (int j = 0; j < THREADS_PER_BLOCK; j++) {
       int gindex = i+j;
       temp[j+RADIUS] = a[gindex]; 
       if (j < RADIUS) {
          temp[j] = (gindex < RADIUS) ? 0 : a[gindex - RADIUS];
          temp[j + RADIUS + BLOCK_SIZE] = a[gindex + BLOCK_SIZE];
       }
     }

     #pragma omp parallel for schedule(static,1)
     for (int j = 0; j < THREADS_PER_BLOCK; j++) {
       int result = 0;
       for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
          result += temp[j+RADIUS+offset];
       b[i+j] = result; 
     }
   }


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
   printf("PASSED\n");
   return 0;
}
