#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "neuron_update.h"

int main(int argc, char* argv[])
{
  int N = atoi(argv[1]);
  int iteration = atoi(argv[2]);
  srand(2);

  float *ge, *gi, *h, *m, *n, *v, *lastspike, *dt, *t;
  ge = gi = h = m = n = v = lastspike = dt = t = NULL;
  char *not_refract = NULL;

  posix_memalign((void**)&ge, 1024, N * sizeof(float));
  posix_memalign((void**)&gi, 1024, N * sizeof(float));
  posix_memalign((void**)&h, 1024,  N * sizeof(float));
  posix_memalign((void**)&m, 1024,  N * sizeof(float));
  posix_memalign((void**)&n, 1024,  N * sizeof(float));
  posix_memalign((void**)&v, 1024,  N * sizeof(float));
  posix_memalign((void**)&lastspike, 1024,  N * sizeof(float));
  posix_memalign((void**)&dt, 1024,  1 * sizeof(float));
  posix_memalign((void**)&t, 1024,  1 * sizeof(float));
  posix_memalign((void**)&not_refract, 1024,  N * sizeof(char));

  printf("initializing ... ");
  for (int i = 1; i < N; i++) {
    ge[i] = 1.0f/ (rand() % 1000 + 1);
    gi[i] = 1.0f/ (rand() % 1000 + 1);
     h[i] = 1.0f/ (rand() % 1000 + 1);
     m[i] = 1.0f/ (rand() % 1000 + 1);
     n[i] = 1.0f/ (rand() % 1000 + 1);
     v[i] = 1.0f/ (rand() % 1000 + 1);
    lastspike[i] = 1.0f/ (rand() % 1000 + 1);
  }

  for (int i = 0; i < 1; i++) { 
    dt[i] = 0.0001;
    t[i] = 0.01;
  }
  printf("done.\n");

  // run the kernel with 'N' neurons and 'iteration' counts
  neurongroup_stateupdater (ge, gi, h, m, n, v, lastspike, dt, t, not_refract, N, iteration);
  
#ifdef DEBUG
  for (int i = 0; i < N; i++) {
    printf("%f %f %f %f %f %f %d\n", 
        ge[i], gi[i], h[i], m[i], n[i], v[i], not_refract[i]);
  }
#endif

  free(ge);
  free(gi);
  free(h);
  free(m);
  free(n);
  free(v);
  free(lastspike);
  free(dt);
  free(t);
  free(not_refract);
  return 0;
}
