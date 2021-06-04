#include <stdio.h>
#include <omp.h>
#include <assert.h>

int main() {
  const int len = 256;
  const int iteration = 1 << 20;
  int d[len];
  for (int i = 0; i < len; i++) d[i] = i;

  #pragma omp target data map(tofrom: d[0:len]) 
  {
    for (int i = 0; i <= iteration; i++) {
      #pragma omp target teams num_teams(1) thread_limit(len)
      { 
        int s[len];
        #pragma omp parallel 
        {
          int t = omp_get_thread_num();
          int tr = len-t-1;
          s[t] = d[t];
          #pragma omp barrier
          d[t] = s[tr];
        }
      }
    }
  }

  for (int i = 0; i < len; i++) assert(d[i] == len-i-1);
  printf("PASS\n");

  return 0;
}
