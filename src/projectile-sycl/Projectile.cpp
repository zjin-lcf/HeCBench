//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <vector>
#include <sycl/sycl.hpp>
#include "Projectile.hpp"

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
void GpuParallel(sycl::queue& q,
                 std::vector<Projectile>& in_vect,
                 std::vector<Projectile>& out_vect,
                 const int repeat)
{
  Projectile *bufin_vect = sycl::malloc_device<Projectile>(num_elements, q);
  q.memcpy(bufin_vect, in_vect.data(), sizeof(Projectile) * num_elements);

  Projectile *bufout_vect = sycl::malloc_device<Projectile>(num_elements, q);

  sycl::range<1> gws ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class projectile>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0); 
        if (i >= num_elements) return;
        float proj_angle = bufin_vect[i].getangle();
        float proj_vel = bufin_vect[i].getvelocity();
        float sin_value = sycl::sin(proj_angle * kPIValue / 180.0f);
        float cos_value = sycl::cos(proj_angle * kPIValue / 180.0f);
        float total_time = sycl::fabs((2 * proj_vel * sin_value)) / kGValue;
        float max_range = sycl::fabs(proj_vel * total_time * cos_value);
        float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                           kGValue;  // h = v^2 * sin^2theta/2g

        bufout_vect[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(out_vect.data(), bufout_vect, sizeof(Projectile) * num_elements).wait();
  sycl::free(bufin_vect, q);
  sycl::free(bufout_vect, q);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  float init_angle = 0.0f;
  float init_vel = 0.0f;
  std::vector<Projectile> input_vect1, out_parallel_vect2, out_scalar_vect3;

  // Initialize the Input and Output vectors
  srand(2);
  for (int i = 0; i < num_elements; i++) {
    init_angle = rand() % 90 + 10;
    init_vel = rand() % 400 + 10;
    input_vect1.push_back(Projectile(init_angle, init_vel, 1.0f, 1.0f, 1.0f));
    out_parallel_vect2.push_back(Projectile());
    out_scalar_vect3.push_back(Projectile());
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  GpuParallel(q, input_vect1, out_parallel_vect2, repeat);
      
#ifdef DEBUG
  for (int i = 0; i < num_elements; i++)
  {
    // Displaying the Parallel computation results.
    std::cout << "Parallel " << out_parallel_vect2[i];
  }
#endif
  return 0;
}
