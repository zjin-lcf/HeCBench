//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <vector>
#include "Projectile.hpp"

using namespace cl::sycl;
using namespace std;

#ifdef DEBUG
static const int num_elements = 100;
#else
static const int num_elements = 10000000;
#endif
const float kPIValue = 3.1415;
const float kGValue = 9.81;
const int BLOCK_SIZE = 256;

// Function to calculate the range, maximum height and total flight time of a projectile
// in_vect and out_vect are the vectors with N Projectile numbers and are inputs to the
// parallel function
void GpuParallel(queue& q, std::vector<Projectile>& in_vect,
                    std::vector<Projectile>& out_vect) {
  buffer<Projectile, 1> bufin_vect(in_vect.data(), range<1>(num_elements));
  buffer<Projectile, 1> bufout_vect(out_vect.data(), range<1>(num_elements));

  range<1> global_work_size ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
  range<1> local_work_size (BLOCK_SIZE);

  for (int i = 0; i < 100; i++) {
    // Submit Command group function object to the queue
    q.submit([&](handler& h) {
      // Input accessors set as read_write mode
      auto obj = bufin_vect.get_access<access::mode::read_write>(h);

      // Output accessor set to discard write mode.
      auto pObj = bufout_vect.get_access<access::mode::discard_write>(h);

      h.parallel_for(nd_range(global_work_size, local_work_size), [=](nd_item<1> item) {
        int i = item.get_global_id(0); 
        float proj_angle = obj[i].getangle();
        float proj_vel = obj[i].getvelocity();
        // for trignometric functions use cl::sycl::sin/cos
        float sin_value = cl::sycl::sin(proj_angle * kPIValue / 180.0f);
        float cos_value = cl::sycl::cos(proj_angle * kPIValue / 180.0f);
        float total_time = cl::sycl::fabs((2 * proj_vel * sin_value)) / kGValue;
        float max_range = cl::sycl::fabs(proj_vel * total_time * cos_value);
        float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                           kGValue;  // h = v^2 * sin^2theta/2g

        pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
      });
    });
  }
  q.wait();
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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  // Call the DpcppParallel with the required inputs and outputs
  GpuParallel(q, input_vect1, out_parallel_vect2);
      
#ifdef DEBUG
  for (int i = 0; i < num_elements; i++)
  {
    // Displaying the Parallel computation results.
    cout << "Parallel " << out_parallel_vect2[i];
  }
#endif
  return 0;
}
