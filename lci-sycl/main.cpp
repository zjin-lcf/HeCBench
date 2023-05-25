#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include "../lci-cuda/tables.h"
#include "kernels.h"
#include <sycl/sycl.hpp>

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

  int size = sizeof(double) * (dimension+1);

  double *c = (double*) malloc (size);
  double *n = (double*) malloc (size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int dftab_size = sizeof(double_fact_table);
  int ftab_size = sizeof(fact_table);

  double *dftab = (double*) sycl::malloc_device(dftab_size, q);
  q.memcpy(dftab, double_fact_table, dftab_size);

  double *ftab = (double*) sycl::malloc_device(ftab_size, q);
  q.memcpy(ftab, fact_table, ftab_size);

  double *d_c = sycl::malloc_device<double>(dimension+1, q);
  double *d_n = sycl::malloc_device<double>(dimension+1, q);

  initial(c, seed);
#ifdef DUMP
  dump(outdata, c, tmin);
#endif

  sycl::range<1> gws (96);
  sycl::range<1> lws (96); // work-group size >= L_max

  float total_time = 0.f;
  for (double t_next = tmin + delta_t; t_next <= tmax; t_next += delta_t)
  {
    q.memcpy(d_c, c, size).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rhs>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        RHS_f(item, dftab, ftab, t_next, d_c, d_n);
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

#ifdef DUMP
    q.memcpy(n, d_n, size).wait();
    dump(outdata, n, t_next);
#endif
    initial(c, ++seed);
  }
  printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);

#ifdef DUMP
  fclose (outdata);
#endif
  sycl::free(d_c, q);
  sycl::free(d_n, q);
  sycl::free(dftab, q);
  sycl::free(ftab, q);
  free(c);
  free(n);

  return 0;
}
