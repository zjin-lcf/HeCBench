#include <stdio.h>
#include <omp.h>

#define N 512
#define ITERATION 100000


template <typename dataType>
void runTest (dataType *in, dataType *out, int n) 
{
  #pragma omp target data map(to: in[0:N]) map(from: out[0:N])
  {
    for (int i = 0; i < ITERATION; i++) {
      #pragma omp target teams num_teams(1) thread_limit(N/2)
      {
        dataType temp[N];
        #pragma omp parallel 
	{
          int thid = omp_get_thread_num();
          int offset = 1;
          temp[2*thid]   = in[2*thid];
          temp[2*thid+1] = in[2*thid+1];
          for (int d = n >> 1; d > 0; d >>= 1) 
          {
            #pragma omp barrier
            if (thid < d) 
            {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              temp[bi] += temp[ai];
            }
            offset *= 2;
          }

          if (thid == 0) temp[n-1] = 0; // clear the last elem
          for (int d = 1; d < n; d *= 2) // traverse down
          {
            offset >>= 1;     
            #pragma omp barrier
            if (thid < d)
            {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              float t = temp[ai];
              temp[ai] = temp[bi];
              temp[bi] += t;
            }
          }
          out[2*thid] = temp[2*thid];
          out[2*thid+1] = temp[2*thid+1];
	}
      }
    }
  }
}

int main() 
{
  float in[N];
  float cpu_out[N];
  float gpu_out[N];
  int error = 0;
  for (int i = 0; i < N; i++) in[i] = (i % 5)+1;
  runTest(in, gpu_out, N); 
  cpu_out[0] = 0;
  if (gpu_out[0] != 0) {
   error++;
   printf("gpu = %f at index 0\n", gpu_out[0]);
  }
  for (int i = 1; i < N; i++) 
  {
    cpu_out[i] = cpu_out[i-1] + in[i-1];
    if (cpu_out[i] != gpu_out[i]) { 
     error++;
     printf("cpu = %f gpu = %f at index %d\n",
     cpu_out[i], gpu_out[i], i);
    }
  }
  if (error == 0) printf("PASS\n");
  return 0; 
}
