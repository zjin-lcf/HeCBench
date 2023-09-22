#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>

// threads per block
#define BS 256

#include "kernel.h"

double LCG_random_double(uint64_t * seed)
{
  const unsigned long m = 9223372036854775808ULL; // 2^63
  const unsigned long a = 2806196910506780709ULL;
  const unsigned long c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }

  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int m = (n + BS - 1) / BS; // number of groups

  int *nlist = (int*) malloc (sizeof(int) * n);
  int *family = (int*) malloc (sizeof(int) * m);
  int *n_neigh = (int*) malloc (sizeof(int) * m);
  double *damage = (double*) malloc (sizeof(double) * m);

  unsigned long seed = 123;
  for (int i = 0; i < n; i++) {
    nlist[i] = (LCG_random_double(&seed) > 0.5) ? 1 : -1;
  }

  for (int i = 0; i < m; i++) {
    int s = 0;
    for (int j = 0; j < BS; j++) {
      s += (nlist[i*BS+j] != -1) ? 1 : 0;
    }
    // non-zero values
    family[i] = s + 1 + s * LCG_random_double(&seed);
  }

  #pragma omp target data map(to: nlist[0:n], family[0:m]) \
                          map(from: n_neigh[0:m], damage[0:m])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) 
      damage_of_node (n, nlist, family, n_neigh, damage);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
  }

  double sum = 0.0;
  for (int i = 0; i < m; i++) sum += damage[i]; 
  printf("Checksum: total damage = %lf\n", sum);

  free(nlist);
  free(family);
  free(n_neigh);
  free(damage);

  return 0;
}
