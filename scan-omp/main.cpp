#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

// scan over N elements
#define N 512

template<typename T>
void verify(const T* ref_out, const T* out, int n)
{
  int error = memcmp(ref_out, out, n * sizeof(T));
  printf("%s\n", error ? "FAIL" : "PASS");
}

// bank conflict aware optimization
#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

template<typename T>
void runTest (const int repeat, bool timing = false)
{
  T in[N];
  T ref_out[N];
  T out[N];

  int n = N;

  for (int i = 0; i < n; i++) in[i] = (i % 5)+1;
  ref_out[0] = 0;
  for (int i = 1; i < n; i++) ref_out[i] = ref_out[i-1] + in[i-1];

  #pragma omp target data map(to: in[0:n]) map(alloc: out[0:n])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams num_teams(1) thread_limit(n/2)
      {
        T temp[N];
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
    if (timing) {
      printf("Element size in bytes is %zu. Average execution time of block scan (w/  bank conflicts): %f (us)\n",
             sizeof(T), (time * 1e-3f) / repeat);
    }
    #pragma omp target update from (out[0:n])
    if (!timing) verify(ref_out, out, n);

    // bcao
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams num_teams(1) thread_limit(n/2)
      {
        T temp[2*N];
        #pragma omp parallel 
        {
          int thid = omp_get_thread_num();
          int a = thid;
          int b = a + (n/2);
          int oa = OFFSET(a);
          int ob = OFFSET(b);

          temp[a + oa] = in[a];
          temp[b + ob] = in[b];

          int offset = 1;
          for (int d = n >> 1; d > 0; d >>= 1) 
          {
            #pragma omp barrier
            if (thid < d) 
            {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              ai += OFFSET(ai);
              bi += OFFSET(bi);
              temp[bi] += temp[ai];
            }
            offset *= 2;
          }

          if (thid == 0) temp[n-1+OFFSET(n-1)] = 0; // clear the last elem
          for (int d = 1; d < n; d *= 2) // traverse down
          {
            offset >>= 1;     
            #pragma omp barrier      
            if (thid < d)
            {
              int ai = offset*(2*thid+1)-1;
              int bi = offset*(2*thid+2)-1;
              ai += OFFSET(ai);
              bi += OFFSET(bi);
              T t = temp[ai];
              temp[ai] = temp[bi];
              temp[bi] += t;
            }
          }
          #pragma omp barrier // required

          out[a] = temp[a + oa];
          out[b] = temp[b + ob];
        }
      }
    }

    end = std::chrono::steady_clock::now();
    auto bcao_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (timing) {
      printf("Element size in bytes is %zu. Average execution time of block scan (w/o bank conflicts): %f (us). ",
             sizeof(T), (bcao_time * 1e-3f) / repeat);
      printf("Reduce the time by %.1f%%\n", (time - bcao_time) * 1.0 / time * 100);
    }
 
    #pragma omp target update from (out[0:n])
    if (!timing) verify(ref_out, out, n);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
    
  for (int i = 0; i < 2; i++) {
    bool timing = i > 0;
    runTest<char>(repeat, timing);
    runTest<short>(repeat, timing);
    runTest<int>(repeat, timing);
    runTest<long>(repeat, timing);
  }

  return 0; 
}
