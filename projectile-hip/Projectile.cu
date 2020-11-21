//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <vector>
#include <hip/hip_runtime.h>
#include "Projectile.hpp"


#ifdef DEBUG
static const int num_elements = 100;
#else
static const int num_elements = 10000000;
#endif
const float kPIValue = 3.1415;
const float kGValue = 9.81;
const int BLOCK_SIZE = 256; 

// Function to calculate the range, maximum height and total flight time of a
// projectile

__global__ void CalculateRange(const Projectile *obj, Projectile *pObj) {  
  
  unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
  float proj_angle = obj[i].getangle();
  float proj_vel = obj[i].getvelocity();
  // for trignometric functions use cl::sycl::sin/cos
  float sin_value = sin(proj_angle * kPIValue / 180.0f);
  float cos_value = cos(proj_angle * kPIValue / 180.0f);
  float total_time = fabs((2 * proj_vel * sin_value)) / kGValue;
  float max_range =  fabs(proj_vel * total_time * cos_value);
  float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                     kGValue;  // h = v^2 * sin^2theta/2g

  pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
}

// in_vect and out_vect are the vectors with N Projectile numbers and are inputs to the
// parallel function
void GpuParallel( std::vector<Projectile>& in_vect, std::vector<Projectile>& out_vect) {
  Projectile *bufin_vect, *bufout_vect;

  hipMalloc((void**)&bufin_vect, sizeof(Projectile) * num_elements);
  hipMalloc((void**)&bufout_vect, sizeof(Projectile) * num_elements);
  hipMemcpy(bufin_vect, in_vect.data(), sizeof(Projectile) * num_elements, hipMemcpyHostToDevice);
  for (int i = 0; i < 100; i++)
    hipLaunchKernelGGL(CalculateRange, 
      dim3((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0, bufin_vect, bufout_vect);

  hipMemcpy(out_vect.data(), bufout_vect, sizeof(Projectile) * num_elements, hipMemcpyDeviceToHost);
  hipFree(bufin_vect);
  hipFree(bufout_vect);
}

int main() {
  srand(2);
  float init_angle = 0.0f;
  float init_vel = 0.0f;
  vector<Projectile> input_vect1, out_parallel_vect2, out_scalar_vect3;
  // Initialize the Input and Output vectors
  for (int i = 0; i < num_elements; i++) {
    init_angle = rand() % 90 + 10;
    init_vel = rand() % 400 + 10;
    input_vect1.push_back(Projectile(init_angle, init_vel, 1.0f, 1.0f, 1.0f));
    out_parallel_vect2.push_back(Projectile());
    out_scalar_vect3.push_back(Projectile());
  }

  // Call the DpcppParallel with the required inputs and outputs
  GpuParallel(input_vect1, out_parallel_vect2);
      
#ifdef DEBUG
  for (int i = 0; i < num_elements; i++)
  {
        // Displaying the Parallel computation results.
        cout << "Parallel " << out_parallel_vect2[i];
  } 
#endif
  return 0;
}
