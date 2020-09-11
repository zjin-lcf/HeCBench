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

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sys/time.h>

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


/*  coalesced GPU implementation of the all-pairs kernel using
    character data types and registers */
void GPUregister(const char *data, int *distance, sycl::nd_item<3> item_ct1) {
  int idx = item_ct1.get_local_id(2);
  int gx = item_ct1.get_group(2);
  int gy = item_ct1.get_group(1);

  for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
    sycl::char4 j = *(sycl::char4 *)(data + i + ATTRIBUTES * gx);
    sycl::char4 k = *(sycl::char4 *)(data + i + ATTRIBUTES * gy);

    /* use a local variable (stored in register) to hold intermediate
       values. This reduces writes to global memory */
    char count = 0;

    if (j.x() ^ k.x())
      count++;
    if (j.y() ^ k.y())
      count++;
    if (j.z() ^ k.z())
      count++;
    if (j.w() ^ k.w())
      count++;

    /* Only one atomic write to global memory */
    sycl::atomic<int>(sycl::global_ptr<int>(distance + INSTANCES * gx + gy))
        .fetch_add(count);
  }
}

/*  coalesced GPU implementation of the all-pairs kernel using
    character data types, registers, and shared memory */
void GPUshared(const char *data, int *distance, sycl::nd_item<3> item_ct1,
               int *dist) {
  int idx = item_ct1.get_local_id(2);
  int gx = item_ct1.get_group(2);
  int gy = item_ct1.get_group(1);

  /* Shared memory is the other major memory (other than registers and
     global). It is used to store values between multiple threads. In
     particular, the shared memory access is defined by the __shared__
     attribute and it is a special area of memory on the GPU
     itself. Because the memory is on the chip, it is a lot faster
     than global memory. Multiple threads can still access it, though,
     provided they are in the same block.
   */

  /* each thread initializes its own location of the shared array */ 
  dist[idx] = 0;

  /* At this point, the threads must be synchronized to ensure that
     the shared array is fully initialized. */
  item_ct1.barrier();

  for(int i = idx*4; i < ATTRIBUTES; i+=THREADS*4) {
    sycl::char4 j = *(sycl::char4 *)(data + i + ATTRIBUTES * gx);
    sycl::char4 k = *(sycl::char4 *)(data + i + ATTRIBUTES * gy);
    char count = 0;

    if (j.x() ^ k.x())
      count++;
    if (j.y() ^ k.y())
      count++;
    if (j.z() ^ k.z())
      count++;
    if (j.w() ^ k.w())
      count++;

    /* Increment shared array */
    dist[idx] += count;
  }

  /* Synchronize threads to make sure all have completed their updates
     of the shared array. Since the distances for each thread are read
     by thread 0 below, this must be ensured. Above, it was not
     necessary because each thread was accessing its own memory
   */
  item_ct1.barrier();

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
}

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  /* host data */
  int *data; 
  char *data_char;
  int *cpu_distance; 
  int *gpu_distance; 

  /* device data */
  char *data_char_device;
  int *distance_device; 

  /* block and grid dimensions */
  sycl::range<3> dimBlock(1, 1, 1);
  sycl::range<3> dimGrid(1, 1, 1);

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

  /* allocate GPU memory */
  dpct::dpct_malloc((void **)&data_char_device,
                    INSTANCES * ATTRIBUTES * sizeof(char));

  dpct::dpct_malloc((void **)&distance_device,
                    INSTANCES * INSTANCES * sizeof(int));

  dpct::dpct_memcpy(data_char_device, data_char,
                    INSTANCES * ATTRIBUTES * sizeof(char),
                    dpct::host_to_device);

  /* specify grid and block dimensions */
  dimBlock[0] = THREADS;
  dimBlock[1] = 1;
  dimGrid[0] = INSTANCES;
  dimGrid[1] = INSTANCES;

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
    dpct::dpct_memcpy(distance_device, gpu_distance,
                      INSTANCES * INSTANCES * sizeof(int),
                      dpct::host_to_device);
    {
      dpct::buffer_t data_char_device_buf_ct0 =
          dpct::get_buffer(data_char_device);
      dpct::buffer_t distance_device_buf_ct1 =
          dpct::get_buffer(distance_device);
      q_ct1.submit([&](sycl::handler &cgh) {
        auto data_char_device_acc_ct0 =
            data_char_device_buf_ct0.get_access<sycl::access::mode::read_write>(
                cgh);
        auto distance_device_acc_ct1 =
            distance_device_buf_ct1.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = dimGrid * dimBlock;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(dimBlock.get(2), dimBlock.get(1),
                                             dimBlock.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              GPUregister((const char *)(&data_char_device_acc_ct0[0]),
                          (int *)(&distance_device_acc_ct1[0]), item_ct1);
            });
      });
    }
    dpct::dpct_memcpy(gpu_distance, distance_device,
                      INSTANCES * INSTANCES * sizeof(int),
                      dpct::device_to_host);
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
    dpct::dpct_memcpy(distance_device, gpu_distance,
                      INSTANCES * INSTANCES * sizeof(int),
                      dpct::host_to_device);
    {
      dpct::buffer_t data_char_device_buf_ct0 =
          dpct::get_buffer(data_char_device);
      dpct::buffer_t distance_device_buf_ct1 =
          dpct::get_buffer(distance_device);
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dist_acc_ct1(sycl::range<1>(128 /*THREADS*/), cgh);
        auto data_char_device_acc_ct0 =
            data_char_device_buf_ct0.get_access<sycl::access::mode::read_write>(
                cgh);
        auto distance_device_acc_ct1 =
            distance_device_buf_ct1.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = dimGrid * dimBlock;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(dimBlock.get(2), dimBlock.get(1),
                                             dimBlock.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              GPUshared((const char *)(&data_char_device_acc_ct0[0]),
                        (int *)(&distance_device_acc_ct1[0]), item_ct1,
                        dist_acc_ct1.get_pointer());
            });
      });
    }
    dpct::dpct_memcpy(gpu_distance, distance_device,
                      INSTANCES * INSTANCES * sizeof(int),
                      dpct::device_to_host);
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
  dpct::dpct_free(data_char_device);
  dpct::dpct_free(distance_device);

  return status;
}


