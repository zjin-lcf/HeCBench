#include <stdio.h>
#include <float.h>
#include <math.h>
#include "constants.h"

void find_min_max(
    const float *__restrict__ u, llint u_size, 
          float *__restrict__ min_u, float *__restrict__ max_u
) {

   float min_val = FLT_MAX;
   float max_val = FLT_MIN;
   #pragma omp parallel for reduction(min: min_val)
   for (llint i = 0; i < u_size; i++) {
     min_val = fmin(u[i], min_val);
   }
   *min_u = min_val;
   #pragma omp parallel for reduction(max: max_val)
   for (llint i = 0; i < u_size; i++) {
     max_val = fmax(u[i], max_val);
   }
   *max_u = max_val;
}
