#include <float.h>
#include <math.h>
#include "data_setup.h"

void target_init(struct grid_t grid, uint nsteps,
                 const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ phi,
                 const float *__restrict__ eta, const float *__restrict__ coefx, const float *__restrict__ coefy,
                 const float *__restrict__ coefz, const float *__restrict__ vp, const float *__restrict__ source)
{
    // Nothing needed
}

void target_finalize(struct grid_t grid, uint nsteps,
                     const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ phi,
                     const float *__restrict__ eta, const float *__restrict__ coefx, const float *__restrict__ coefy,
                     const float *__restrict__ coefz, const float *__restrict__ vp, const float *__restrict__ source)
{
    // Nothing needed
}

void kernel_add_source(struct grid_t grid,
                       float *__restrict__ u, const float *__restrict__ source, llint istep,
                       llint sx, llint sy, llint sz)
{
    // Nothing needed
}

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
