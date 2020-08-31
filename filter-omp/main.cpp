#include <cstdio>
#include <cstdlib>
#include <omp.h>

#define NUM_ELEMS 10000000
#define NUM_THREADS_PER_BLOCK 256


int main(int argc, char **argv) {
  int *data_to_filter, *filtered_data;
  int nres[1] = {0};

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));
  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate input data.
  for (int i = 0; i < NUM_ELEMS; i++) {
    data_to_filter[i] = rand() % 20;
  }

  int n = NUM_ELEMS;

#pragma omp target data map(to: data_to_filter[0:NUM_ELEMS]) \
                        map(tofrom: nres[0:1]) \
                        map(from: filtered_data[0:NUM_ELEMS])
{
  #pragma omp target teams \
  num_teams((NUM_ELEMS+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK) \
  thread_limit(NUM_THREADS_PER_BLOCK) 
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

  int *host_filtered_data =
      reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nFilter using shared memory %s \n",
         host_flt_count == nres[0] ? "PASSED" : "FAILED");

  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
}
