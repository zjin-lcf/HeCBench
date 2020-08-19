//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <vector>
#include <cstdlib>
#include "Projectile.hpp"

using namespace std;

#ifdef DEBUG
static const int num_elements = 100;
#else
static const int num_elements = 10000000;
#endif
const float kPIValue = 3.1415;
const float kGValue = 9.81;
const int BLOCK_SIZE = 256;

// in_vect and out_vect are the vectors with N Projectile numbers and are inputs to the
// parallel function
void GpuParallel( std::vector<Projectile>& in_vect, std::vector<Projectile>& out_vect) {
  Projectile *obj = in_vect.data();
  Projectile *pObj = out_vect.data();

  //for (int i = 0; i < 100; i++)
  // Submit Command group function object to the queue
  #pragma omp target data map(to: obj[0:num_elements]) map(from: pObj[0:num_elements])
  {
    #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
    for (int i = 0; i < num_elements; i++) {
      float proj_angle = obj[i].getangle();
      float proj_vel = obj[i].getvelocity();
      // for trignometric functions use cl::sycl::sin/cos
      float sin_value = sinf(proj_angle * kPIValue / 180.0f);
      float cos_value = cosf(proj_angle * kPIValue / 180.0f);
      float total_time =fabsf((2 * proj_vel * sin_value)) / kGValue;
      float max_range = fabsf(proj_vel * total_time * cos_value);
      float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                         kGValue;  // h = v^2 * sin^2theta/2g
      pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
    }
  }
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
