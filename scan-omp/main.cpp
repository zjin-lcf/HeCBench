#include <stdio.h>
#include <chrono>
#include <omp.h>

#define N 512

template <typename dataType>
void runTest (const dataType *in, dataType *out, const int n, const int repeat) 
{
  #pragma omp target data map(to: in[0:n]) map(from: out[0:n])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams num_teams(1) thread_limit(n/2)
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

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of block scan: %f (us)\n", (time * 1e-3f) / repeat);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
    
  float in[N];
  float cpu_out[N];
  float gpu_out[N];
  for (int i = 0; i < N; i++) in[i] = (i % 5)+1;

  runTest(in, gpu_out, N, repeat);

  bool ok = true;
  if (gpu_out[0] != 0) {
    ok = false;
  }

  cpu_out[0] = 0;
  for (int i = 1; i < N; i++) 
  {
    cpu_out[i] = cpu_out[i-1] + in[i-1];
    if (cpu_out[i] != gpu_out[i]) { 
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0; 
}
