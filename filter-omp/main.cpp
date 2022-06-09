#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  int *data_to_filter, *filtered_data;
  int nres[1];

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));
  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));

  // Generate input data.
  srand(2);
  for (int i = 0; i < num_elems; i++) {
    data_to_filter[i] = rand() % 20;
  }

  int n = num_elems;

  #pragma omp target data map(to: data_to_filter[0:num_elems]) \
                          map(from: nres[0:1]) \
                          map(from: filtered_data[0:num_elems])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      nres[0] = 0;
      #pragma omp target update to (nres[0:1]) // need to reset on device 

      #pragma omp target teams num_teams((num_elems+block_size-1)/block_size) \
      thread_limit(block_size) 
      {
        int l_n;
        #pragma omp parallel 
        {
          int i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num() ;
          if (omp_get_thread_num() == 0)
            l_n = 0;
          #pragma omp barrier
          int d, pos;
        
          if(i < n) {
            d = data_to_filter[i];
            if(d > 0) {
              #pragma omp atomic capture
              pos = l_n++;
            }
          }
          #pragma omp barrier
  
          // leader increments the global counter
          if (omp_get_thread_num() == 0) {
            //l_n = atomicAdd(nres, l_n);
             int old;
             #pragma omp atomic capture
             {
                old = nres[0];
                nres[0] += l_n; 
             }
             l_n = old;
          }
          #pragma omp barrier
        
          // threads with true predicates write their elements
          if(i < n && d > 0) {
            pos += l_n; // increment local pos by global counter
            filtered_data[pos] = d;
          }
          #pragma omp barrier
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
  }

  int *host_filtered_data =
      reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nFilter using shared memory %s \n",
         host_flt_count == nres[0] ? "PASS" : "FAIL");

  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
  return 0;
}
