//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
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

void CalculateRange(const Projectile *obj, Projectile *pObj,
                    sycl::nd_item<3> item_ct1) {

  unsigned int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                   item_ct1.get_local_id(2);
  float proj_angle = obj[i].getangle();
  float proj_vel = obj[i].getvelocity();
  // for trignometric functions use cl::sycl::sin/cos
  float sin_value = sycl::sin(proj_angle * kPIValue / 180.0f);
  float cos_value = sycl::cos(proj_angle * kPIValue / 180.0f);
  float total_time = sycl::fabs((2 * proj_vel * sin_value)) / kGValue;
  float max_range = sycl::fabs(proj_vel * total_time * cos_value);
  float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                     kGValue;  // h = v^2 * sin^2theta/2g

  pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
}

// in_vect and out_vect are the vectors with N Projectile numbers and are inputs to the
// parallel function
void GpuParallel( std::vector<Projectile>& in_vect, std::vector<Projectile>& out_vect) {
  Projectile *bufin_vect, *bufout_vect;

  dpct::dpct_malloc((void **)&bufin_vect, sizeof(Projectile) * num_elements);
  dpct::dpct_malloc((void **)&bufout_vect, sizeof(Projectile) * num_elements);
  dpct::dpct_memcpy(bufin_vect, in_vect.data(),
                    sizeof(Projectile) * num_elements, dpct::host_to_device);
  for (int i = 0; i < 100; i++)
  {
    dpct::buffer_t bufin_vect_buf_ct0 = dpct::get_buffer(bufin_vect);
    dpct::buffer_t bufout_vect_buf_ct1 = dpct::get_buffer(bufout_vect);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto bufin_vect_acc_ct0 =
          bufin_vect_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto bufout_vect_acc_ct1 =
          bufout_vect_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1,
                             (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                  sycl::range<3>(1, 1, BLOCK_SIZE),
              sycl::range<3>(1, 1, BLOCK_SIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            CalculateRange((const Projectile *)(&bufin_vect_acc_ct0[0]),
                           (Projectile *)(&bufout_vect_acc_ct1[0]), item_ct1);
          });
    });
  }

  dpct::dpct_memcpy(out_vect.data(), bufout_vect,
                    sizeof(Projectile) * num_elements, dpct::device_to_host);
  dpct::dpct_free(bufin_vect);
  dpct::dpct_free(bufout_vect);
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
