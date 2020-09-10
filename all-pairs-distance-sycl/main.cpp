/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.

   Please feel free to use, modify, or redistribute this code.

   Make sure you have a CUDA compatible GPU and the nvcc is installed.
   To compile, type make.
   After compilation, type ./chapter to run
   Output written to timing.txt
   */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include "common.h"

#define INSTANCES 224   /* # of instances */
#define ATTRIBUTES 4096 /* # of attributes */
#define THREADS 128    /* # of threads per block */

/* CPU implementation */
void CPU(int * data, int * distance) {
  /* compare all pairs of instances, accessing the attributes in
     row-major order */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] += 
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}



int main(int argc, char **argv) {

  /* host data */
  int *data; 
  char *data_char;
  int *cpu_distance; 
  int *gpu_distance; 

  /* used to time CPU and GPU implementations */
  double start_cpu, stop_cpu;
  double start_gpu, stop_gpu;
  float elapsedTime; 
  struct timeval tp;
  struct timezone tzp;
  /* verification result */ 
  int status;

  /* output file for timing results */
  FILE *out = fopen("timing.txt","a");

  /* seed RNG */
  srand(2);

  /* allocate host memory */
  data = (int *)malloc(INSTANCES * ATTRIBUTES * sizeof(int));
  data_char = (char *)malloc(INSTANCES * ATTRIBUTES * sizeof(char));
  cpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));
  gpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));

  /* randomly initialize host data */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < ATTRIBUTES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
    }
  }

#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  /* allocate GPU memory */
  buffer<char, 1> data_char_device(data_char, INSTANCES * ATTRIBUTES);
  buffer<int, 1> distance_device(INSTANCES * INSTANCES);

  range<2> global_size (INSTANCES, INSTANCES*THREADS);
  range<2> local_size (1, THREADS);

  /* CPU */
  bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
  gettimeofday(&tp, &tzp);
  start_cpu = tp.tv_sec*1000000+tp.tv_usec;
  CPU(data, cpu_distance);
  gettimeofday(&tp, &tzp);
  stop_cpu = tp.tv_sec*1000000+tp.tv_usec;
  elapsedTime = stop_cpu - start_cpu;
  fprintf(out,"%f ",elapsedTime);

  for (int n = 0; n < 10; n++) {
    /* register GPU kernel */
    bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;

    q.submit([&] (handler &h) {
        auto distance_acc = distance_device.get_access<sycl_write>(h);
        h.copy(gpu_distance, distance_acc);
        });

    q.submit([&] (handler &h) {
        auto data = data_char_device.get_access<sycl_read>(h);
        auto distance = distance_device.get_access<sycl_atomic>(h);
        h.parallel_for<class GPUregister>(nd_range<2> (global_size, local_size), [=] (nd_item<2> item) {
            int idx = item.get_local_id(1); //threadIdx.x;
            int gx = item.get_group(1); //blockIdx.x;
            int gy = item.get_group(0); //blockIdx.y;

            vec<char, 4> j, k;
            for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
              j.load(i/4, data.get_pointer() + ATTRIBUTES*gx);
              k.load(i/4, data.get_pointer() + ATTRIBUTES*gy);

            /* use a local variable (stored in register) to hold intermediate
               values. This reduces writes to global memory */
              int count = 0;

              if(j.x() ^ k.x()) 
                count++; 
              if(j.y() ^ k.y())
                count++;
              if(j.z() ^ k.z())
                count++;
              if(j.w() ^ k.w())
                count++;

              /* Only one atomic write to global memory */
              atomic_fetch_add(distance[INSTANCES*gx + gy], count);
            }
        });
    });
    q.submit([&] (handler &h) {
        auto distance_acc = distance_device.get_access<sycl_read>(h);
        h.copy(distance_acc, gpu_distance);
        });
    q.wait();

    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime = stop_gpu - start_gpu;
    fprintf(out,"%f ",elapsedTime);
  }

  status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status != 0) printf("FAIL\n");
  else printf("PASS\n");

  for (int n = 0; n < 10; n++) {
    /* shared memory GPU kernel */
    bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;

    q.submit([&] (handler &h) {
        auto distance_acc = distance_device.get_access<sycl_write>(h);
        h.copy(gpu_distance, distance_acc);
        });

    /*  coalesced GPU implementation of the all-pairs kernel using
        character data types, registers, and shared memory */
    q.submit([&] (handler &h) {
        auto data = data_char_device.get_access<sycl_read>(h);
        auto distance = distance_device.get_access<sycl_read_write>(h);
        accessor<int, 1, sycl_read_write, access::target::local> dist(THREADS, h); 
        h.parallel_for<class GPUshared>(nd_range<2> (global_size, local_size), [=] (nd_item<2> item) {
            int idx = item.get_local_id(1); //threadIdx.x;
            int gx = item.get_group(1); //blockIdx.x;
            int gy = item.get_group(0); //blockIdx.y;

            /* each thread initializes its own location of the shared array */ 
            dist[idx] = 0;

            /* At this point, the threads must be synchronized to ensure that
               the shared array is fully initialized. */
            item.barrier(access::fence_space::local_space);

            vec<char, 4> j, k;
            for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
              j.load(i/4, data.get_pointer() + ATTRIBUTES*gx);
              k.load(i/4, data.get_pointer() + ATTRIBUTES*gy);

            /* use a local variable (stored in register) to hold intermediate
               values. This reduces writes to global memory */
              char count = 0;
              if(j.x() ^ k.x()) 
                count++; 
              if((j.y() ^ k.y())
                count++;
              if((j.z() ^ k.z())
                count++;
              if((j.w() ^ k.w())
                count++;

              /* Increment shared array */
              dist[idx] += count;
            }

            /* Synchronize threads to make sure all have completed their updates
               of the shared array. Since the distances for each thread are read
               by thread 0 below, this must be ensured. Above, it was not
               necessary because each thread was accessing its own memory
               */
            item.barrier(access::fence_space::local_space);

            /* Reduction: Thread 0 will add the value of all other threads to
               its own */ 
            if(idx == 0) {
              for(int i = 1; i < THREADS; i++) {
                dist[0] += dist[i];
              }

              /* Thread 0 will then write the output to global memory. Note that
                 this does not need to be performed atomically, because only one
                 thread per block is writing to global memory, and each block
                 corresponds to a unique memory address. 
                 */
              distance[INSTANCES*gy + gx] = dist[0];
            }
        });
    });
    q.submit([&] (handler &h) {
        auto distance_acc = distance_device.get_access<sycl_read>(h);
        h.copy(distance_acc, gpu_distance);
        });
    q.wait();
    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime = stop_gpu - start_gpu;
    fprintf(out,"%f ",elapsedTime);
  }
  status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status != 0) printf("FAIL\n");
  else printf("PASS\n");

  fclose(out);
  free(cpu_distance);
  free(gpu_distance);
  free(data);

  return status;
}


