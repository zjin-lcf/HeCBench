/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#define INSTANCES 224   /* # of instances */
#define ATTRIBUTES 4096 /* # of attributes */
#define THREADS 128    /* # of threads per block */

struct char4 { char x; char y; char z; char w; };


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

  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]);

  /* host data */
  int *data;
  char *data_char;
  int *cpu_distance, *gpu_distance;

  /* verification result */
  int status;

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

  /* CPU */
  auto start = std::chrono::steady_clock::now();
  bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
  CPU(data, cpu_distance);
  auto end = std::chrono::steady_clock::now();
  double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  printf("CPU time: %f (us)\n", elapsedTime);

  #pragma omp target data map(to: data_char[0:INSTANCES * ATTRIBUTES]) \
                          map(alloc: gpu_distance[0:INSTANCES * INSTANCES ])
  {
    start = std::chrono::steady_clock::now();
    for (int n = 0; n < iterations; n++) {
      /* register-based kernel */
      #pragma omp target teams distribute parallel for nowait
      for (int i = 0; i < INSTANCES*INSTANCES; i++) {
        gpu_distance[i] = 0;
      }

      #pragma omp target teams num_teams(INSTANCES*INSTANCES) thread_limit(THREADS)
      {
        #pragma omp parallel
        {
          int idx = omp_get_thread_num();
          int gx = omp_get_team_num() % INSTANCES;
          int gy = omp_get_team_num() / INSTANCES;

          for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
            char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
            char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);

            /* use a local variable (stored in register) to hold intermediate
               values. This reduces writes to global memory */
            char count = 0;

            if(j.x ^ k.x)
              count++;
            if(j.y ^ k.y)
              count++;
            if(j.z ^ k.z)
              count++;
            if(j.w ^ k.w)
              count++;

            /* atomic write to global memory */
            #pragma omp atomic update
            gpu_distance[ INSTANCES*gx + gy ] += count;
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    #pragma omp target update from (gpu_distance[0:INSTANCES * INSTANCES])

    printf("Average kernel execution time (w/o shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    printf("%s\n", status ? "FAIL" : "PASS");

    start = std::chrono::steady_clock::now();
    for (int n = 0; n < iterations; n++) {
      /* shared memory GPU kernel */
      #pragma omp target teams distribute parallel for nowait
      for (int i = 0; i < INSTANCES*INSTANCES; i++) {
        gpu_distance[i] = 0;
      }

      #pragma omp target teams num_teams(INSTANCES*INSTANCES) thread_limit(THREADS)
      {
        int dist[THREADS];
        #pragma omp parallel
        {
          int idx = omp_get_thread_num();
          int gx = omp_get_team_num() % INSTANCES;
          int gy = omp_get_team_num() / INSTANCES;

          dist[idx] = 0;
          #pragma omp barrier

          for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
            char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
            char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);

            /* use a local variable (stored in register) to hold intermediate
               values. This reduces writes to global memory */
            char count = 0;

            if(j.x ^ k.x)
              count++;
            if(j.y ^ k.y)
              count++;
            if(j.z ^ k.z)
              count++;
            if(j.w ^ k.w)
              count++;

            dist[idx] += count;
          }

        /* Synchronize threads to make sure all have completed their updates
           of the shared array. Since the distances for each thread are read
           by thread 0 below, this must be ensured. Above, it was not
           necessary because each thread was accessing its own memory
        */
          #pragma omp barrier

          /* Perform balanced tree reduction across the shared memory */
          for (int stride = THREADS/2; stride > 0; stride /= 2) {
            if (idx < stride) {
              dist[idx] += dist[idx + stride];
            }
            #pragma omp barrier
          }

            /* Thread 0 will then write the output to global memory. Note that
               this does not need to be performed atomically, because only one
               thread per block is writing to global memory, and each block
               corresponds to a unique memory address.
            */
          if(idx == 0) {
            gpu_distance[INSTANCES*gy + gx] = dist[0];
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    #pragma omp target update from (gpu_distance[0:INSTANCES * INSTANCES])

    printf("Average kernel execution time (w/ shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    printf("%s\n", status ? "FAIL" : "PASS");
  }

  free(cpu_distance);
  free(gpu_distance);
  free(data_char);
  free(data);
  return status;
}
