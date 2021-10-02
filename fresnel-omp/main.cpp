#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#pragma omp declare target
double Fresnel_Sine_Integral(double);
#pragma omp end declare target


void reference (const double *__restrict input, double *__restrict output, const int n) {
  for (int i = 0; i < n; i++)
    output[i] = Fresnel_Sine_Integral(input[i]);
}

int main() {
  // range [0, 8], interval 1e-7
  const double interval = 1e-7;
  const int points = (int)(8.0 / interval);
  double *x = (double*) malloc (sizeof(double) * points);
  double *output = (double*) malloc (sizeof(double) * points);
  double *h_output = (double*) malloc (sizeof(double) * points);
  for (int i = 0; i < points; i++)
    x[i] = (double)i * interval;
	   
#pragma omp target data map(x[0:points]) map(from: output[0:points])
{
  for (int i = 0; i < 100; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < points; i++) 
      output[i] = Fresnel_Sine_Integral(x[i]);
  }
}

  // verify
  reference(x, h_output, points);
  bool ok = true;
  for (int i = 0; i < points; i++) {
    if (fabs(h_output[i] - output[i]) > 1e-6) {
      printf("%lf %lf\n", h_output[i], output[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  
  free(x);
  free(output);
  free(h_output);
  return 0;
}
