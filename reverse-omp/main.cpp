#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: ./%s <iterations>\n", argv[0]);
    return 1;
  }

  // specify the number of test cases
  const int iteration = atoi(argv[1]);

  // number of elements to reverse
  const int len = 256;
  const int elem_size = len * sizeof(int);

  // device result
  int test[len];

  // expected results after reverse operations even/odd times
  int error = 0;
  int gold_odd[len];
  int gold_even[len];

  for (int i = 0; i < len; i++) {
    gold_odd[i] = len-i-1;
    gold_even[i] = i;
  }

  #pragma omp target data map(alloc: test[0:len]) 
  {
    srand(123);
    for (int i = 0; i < iteration; i++) {

      const int count = rand() % 10000 + 100;  // bound the reverse range

      memcpy(test, gold_even, elem_size);
      #pragma omp target update to (test[0:len])

      for (int j = 0; j < count; j++) {
        #pragma omp target teams num_teams(1) thread_limit(len)
        {
          int s[len];
          #pragma omp parallel 
          {
            int t = omp_get_thread_num();
            s[t] = test[t];
            #pragma omp barrier
            test[t] = s[len-t-1];
          }
        }
      }

      #pragma omp target update from (test[0:len])

      if (count % 2 == 0)
        error = memcmp(test, gold_even, elem_size);
      else
        error = memcmp(test, gold_odd, elem_size);
      
      if (error) break;
    }
  }

  printf("%s\n", error ? "FAIL" : "PASS");

  return 0;
}
