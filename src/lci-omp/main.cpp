#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "../lci-cuda/tables.h"
#include "kernels.h"

const double t_init = .1;
const double t_final = 200;

void initial(double c[], int seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dis(-6.0, 6.0);
  std::uniform_real_distribution<double> temp(0.1, 0.9);
  for(int l = 1; l < L_max; l++) c[l] = dis(gen);
  c[0] = temp(gen) ; // random temperature
  c[L_max] = 0.0; //truncation; do not modify
}

#ifdef DUMP
void dump (FILE * out, double c[], double t)
{
  fprintf(out,"%.5e ", t);
  int L = (L_max > 4) ? 4 : L_max; // print a subset 
  for(int l = 0; l < L; l++)
  {
    fprintf(out,"%.5e ", c[l]);
  }
  fprintf(out,"\n");
  fflush(out);
}
#endif

int main (int argc, char* argv[]) {

#ifdef DUMP
  FILE * outdata = fopen ("output.csv","w");
  if (outdata == NULL) {
    printf("Failed to open file for write\n");
    return 1;
  }
#endif

  double tmin = t_init;
  double tmax = t_final;
  double delta_t = 0.1;

  int dimension = L_max;
  int seed = dimension;

  double *c, *n;
  c = (double*) malloc (sizeof(double) * (dimension+1));
  n = (double*) malloc (sizeof(double) * (dimension+1));

  int dftab_size = sizeof(double_fact_table) / sizeof(double_fact_table[0]);
  int ftab_size = sizeof(fact_table) / sizeof(fact_table[0]);

  #pragma omp target data map (alloc: c[0:dimension+1], n[0:dimension+1]) \
                          map (to: fact_table[0:ftab_size], double_fact_table[0:dftab_size])
  {
    initial(c, seed);
    #ifdef DUMP
    dump(outdata,c,tmin);
    #endif
  
    float total_time = 0.f;
    for (double t_next = tmin + delta_t; t_next <= tmax; t_next += delta_t)
    {
      #pragma omp target update to (c[0:dimension+1])

      auto start = std::chrono::steady_clock::now();

      RHS_f (double_fact_table, fact_table, t_next, c, n);

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      total_time += time;

      #ifdef DUMP
      #pragma omp target update from (n[0:dimension+1])
      dump(outdata,n,t_next);
      #endif

      initial(c, ++seed);
    }
    printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);

    #ifdef DUMP
      fclose (outdata);
    #endif
  }

  free(c);
  free(n);

  return 0;
}
