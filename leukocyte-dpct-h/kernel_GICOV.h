#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void kernel_GICOV(const float *grad_x, const float *grad_y,
                  const float *sin_angle, const float *cos_angle, const int *tX,
                  const int *tY, float *gicov, const int local_work_size,
                  const int num_work_groups, const int grad_m,
                  sycl::nd_item<3> item_ct1)
{
  int i, j, k, n, x, y;
  int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);

  if (gid >= local_work_size*num_work_groups) return;

  // Determine this thread's pixel
  i = gid/local_work_size + MAX_RAD + 2;
  j = gid%local_work_size + MAX_RAD + 2;

  // Initialize the maximal GICOV score to 0
  float max_GICOV = 0.f;

  // Iterate across each stencil
  for (k = 0; k < NCIRCLES; k++) {
    // Variables used to compute the mean and variance
    //  of the gradients along the current stencil
    float sum = 0.f, M2 = 0.f, mean = 0.f;    

    // Iterate across each sample point in the current stencil
    for (n = 0; n < NPOINTS; n++) {
      // Determine the x- and y-coordinates of the current sample point
      y = j + tY[(k * NPOINTS) + n];
      x = i + tX[(k * NPOINTS) + n];

      // Compute the combined gradient value at the current sample point
      int addr = x * grad_m + y;
      float p = grad_x[addr] * cos_angle[n] + 
        grad_y[addr] * sin_angle[n];

      // Update the running total
      sum += p;

      // Partially compute the variance
      float delta = p - mean;
      mean = mean + (delta / (float) (n + 1));
      M2 = M2 + (delta * (p - mean));
    }

    // Finish computing the mean
    mean = sum / ((float) NPOINTS);

    // Finish computing the variance
    float var = M2 / ((float) (NPOINTS - 1));

    // Keep track of the maximal GICOV value seen so far
    if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
  }

  // Store the maximal GICOV value
  gicov[(i * grad_m) + j] = max_GICOV;
}
