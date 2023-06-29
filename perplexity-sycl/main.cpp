/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <sycl/sycl.hpp>

#include "reference.cpp"

// forward declaration
template <typename value_idx, typename value_t>
class search;

// Find the best Gaussian bandwidth for each row in the dataset
template <typename value_idx, typename value_t>
void perplexity_search(sycl::queue &q,
                       const value_t *d_distances,
                       value_t *d_data,
                       const float perplexity,
                       const int epochs,
                       const float tol,
                       const value_idx n,
                       const int k,
                       double &time)
{
  const float desired_entropy = logf(perplexity);
  sycl::range<1> gws ((n+255)/256*256);
  sycl::range<1> lws  (256);

  auto start = std::chrono::steady_clock::now();

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class search<value_idx, value_t>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      // For every item in row
      const int i = item.get_global_id(0);
      if (i >= n) return;

      value_t beta_min = -INFINITY, beta_max = INFINITY;
      value_t beta = 1;
      const int ik = i * k;
      int step;

      for (step = 0; step < epochs; step++) {
        value_t sum = FLT_EPSILON;

        // Exponentiate to get Gaussian
        for (int j = 0; j < k; j++) {
          d_data[ik + j] = sycl::native::exp(-d_distances[ik + j] * beta);
          sum += d_data[ik + j];
        }

        // Normalize
        value_t sum_dist = 0;
        const value_t div    = sycl::native::divide(1.0f, sum);
        for (int j = 0; j < k; j++) {
          d_data[ik + j] *= div;
          sum_dist += d_distances[ik + j] * d_data[ik + j];
        }

        const value_t entropy      = sycl::native::log(sum) + beta * sum_dist;
        const value_t entropy_diff = entropy - desired_entropy;
        if (sycl::fabs(entropy_diff) <= tol) {
          break;
        }

        // Bisection search
        if (entropy_diff > 0) {
          beta_min = beta;
          if (sycl::isinf(beta_max))
            beta *= 2.0f;
          else
            beta = (beta + beta_max) * 0.5f;
        } else {
          beta_max = beta;
          if (sycl::isinf(beta_min))
            beta *= 0.5f;
          else
            beta = (beta + beta_min) * 0.5f;
        }
      }
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points> <perplexity> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]); // points
  const int p = atoi(argv[2]); // perplexity
  const int repeat = atoi(argv[3]);

  const int n_nbrs = 4 * p;    // neighbors
  const int max_iter = 100;    // maximum number of iterations
  const float tol = 1e-8f;     // tolerance

  srand(123);
  std::vector<float> data(n * n_nbrs);
  std::vector<float> h_data(n * n_nbrs);
  std::vector<float> distance(n * n_nbrs);
  for (int i = 0; i < n * n_nbrs; i++) {
    distance[i] = rand() / (float)RAND_MAX;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_data = sycl::malloc_device<float>(n*n_nbrs, q);

  float *d_distance = sycl::malloc_device<float>(n*n_nbrs, q);
  q.memcpy(d_distance, distance.data(), sizeof(float)*n*n_nbrs).wait();

  double time = 0.0;

  for (int i = 0; i < repeat; i++)
    perplexity_search(q, d_distance, d_data, p, max_iter, tol, n, n_nbrs, time);

  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(data.data(), d_data, sizeof(float)*n*n_nbrs).wait();

  // verify
  reference(distance.data(), h_data.data(), p, max_iter, tol, n, n_nbrs);

  bool ok = true;
  for (int i = 0; i < n*n_nbrs; i++) {
    if (fabsf(data[i] - h_data[i]) > 1e-3f) {
      printf("%d %f %f\n", i, data[i], h_data[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_distance, q);
  sycl::free(d_data, q);
  return 0;
}

