#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "linear.h"
#include "kernel.h"

#define local_size TEMP_WORKGROUP_SIZE

int cpu_offset;

void linear_regressionOMP(data_t* dataset, sum_t* result, int gpu_size, int global_size)
{
  sum_t* interns = (sum_t*)malloc(sizeof(sum_t) * local_size);

  for (int group_id = (gpu_size / local_size); group_id < (global_size / local_size); group_id++) {
    for (int loc_id = local_size - 1; loc_id >= 0; loc_id--) {
      int glob_id = loc_id + group_id * local_size;
      interns[loc_id].x() = dataset[glob_id].x();
      interns[loc_id].y() = dataset[glob_id].y();
      interns[loc_id].z() = (dataset[glob_id].x() * dataset[glob_id].y());
      interns[loc_id].w() = (dataset[glob_id].x() * dataset[glob_id].x());
      for (int i = (local_size / 2), old_i = local_size; i > 0; old_i = i, i /= 2) {
        if (loc_id < i) {
          interns[loc_id].x() += interns[loc_id + i].x();
          interns[loc_id].y() += interns[loc_id + i].y();
          interns[loc_id].z() += interns[loc_id + i].z();
          interns[loc_id].w() += interns[loc_id + i].w();
          if (loc_id == (i - 1) && old_i % 2 != 0) {
            interns[loc_id].x() += interns[old_i - 1].x();
            interns[loc_id].y() += interns[old_i - 1].y();
            interns[loc_id].z() += interns[old_i - 1].z();
            interns[loc_id].w() += interns[old_i - 1].w();
          }
        }
      }
      if (loc_id == 0) result[group_id] = interns[0];
    }
  }

  free(interns);
}

void rsquaredOMP(
    data_t* dataset,
    float mean,
    sycl::float2 equation,  // [a0,a1]
    rsquared_t* result,
    int gpu_size,
    int global_size)
{
  rsquared_t* dist = (rsquared_t*)malloc(sizeof(rsquared_t) * local_size);

  for (int group_id = (gpu_size / local_size); group_id < (global_size / local_size); group_id++) {
    for (int loc_id = local_size - 1; loc_id >= 0; loc_id--) {
      int glob_id = loc_id + group_id * local_size;
      dist[loc_id].x() = powf((dataset[glob_id].x() - mean), 2.f);
      float y_estimated = dataset[glob_id].x() * equation.x() + equation.y();
      dist[loc_id].y() = powf((y_estimated - mean), 2.f);
      for (int i = (local_size / 2), old_i = local_size; i > 0; old_i = i, i /= 2) {
        if (loc_id < i) {
          dist[loc_id].x() += dist[loc_id + i].x();
          dist[loc_id].y() += dist[loc_id + i].y();
          if (loc_id == (i - 1) && old_i % 2 != 0) {
            dist[loc_id].x() += dist[old_i - 1].x();
            dist[loc_id].y() += dist[old_i - 1].y();
          }
        }
      }
      if (loc_id == 0) {
        result[group_id] = dist[0];
      }
    }
  }

  free(dist);
}

static
void r_squared(sycl::queue &q, linear_param_t *params, data_t *dataset, sum_t *linreg, result_t *response)
{
  float mean = linreg->y() / params->size;
  sycl::float2 equation = {response->a0, response->a1};

  const size_t wg_count = params->wg_count;
  const size_t wg_size = params->wg_size;
  const size_t size = params->size;

  rsquared_t *results = (rsquared_t*) malloc(sizeof(rsquared_t) * wg_count);

  size_t globalWorkSize = size;
  if (size % wg_size) globalWorkSize += wg_size - (size % wg_size);

  size_t cpu_global_size = cpu_offset * globalWorkSize / 100;
  if (cpu_global_size % wg_size != 0) {
    cpu_global_size = (1 + cpu_global_size / wg_size) * wg_size;
  }

  size_t gpu_global_size = globalWorkSize - cpu_global_size;

  /* Create data buffer */
  data_t *d_dataset = nullptr;
  rsquared_t *d_result = nullptr;

  if (gpu_global_size > 0) {
    d_dataset = sycl::malloc_device<data_t>(size, q);
    q.memcpy(d_dataset, dataset, sizeof(data_t) * size);
    d_result = sycl::malloc_device<rsquared_t>(wg_count, q);

    sycl::range<1> gws (gpu_global_size);
    sycl::range<1> lws (wg_size);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < params->repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<rsquared_t, 1> sm (sycl::range<1>(wg_size), cgh);
        cgh.parallel_for<class rs>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          rsquared(item, d_dataset, mean, equation, sm.get_pointer(), d_result);
        });
      });
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    response->ktime += time;
  }

  if (cpu_offset > 0) {
    rsquaredOMP(dataset, mean, equation, results, gpu_global_size, params->size);
  }

  if (gpu_global_size > 0) {
    q.memcpy(results, d_result, sizeof(rsquared_t) * wg_count * (100-cpu_offset) / 100).wait();
  }

  rsquared_t final_result = {0.f, 0.f};

  for (size_t i = 0; i < wg_count; i++) {
    final_result.x() += results[i].x();
    final_result.y() += results[i].y();
  }

  response->rsquared = final_result.y() / final_result.x() * 100;

  free(results);
  if (gpu_global_size > 0) {
    sycl::free(d_dataset, q);
    sycl::free(d_result, q);
  }
}

void parallelized_regression(linear_param_t *params, data_t *dataset, result_t *response)
{
  const size_t size = params->size;
  const size_t wg_size = params->wg_size;
  const size_t wg_count = params->wg_count;

  /* Data and buffers */
  sum_t *results = (sum_t *) malloc(sizeof(sum_t) * wg_count);

  size_t globalWorkSize = size;
  if (size % wg_size) globalWorkSize += wg_size - (size % wg_size);

  size_t cpu_global_size;
  cpu_global_size = cpu_offset * globalWorkSize / 100;
  if (cpu_global_size % wg_size != 0) {
    cpu_global_size = (1 + cpu_global_size / wg_size) * wg_size;
  }

  size_t gpu_global_size;
  gpu_global_size = globalWorkSize - cpu_global_size;

  data_t *d_dataset = nullptr;
  sum_t *d_result = nullptr;
  sycl::queue *d_q = nullptr;

  if (gpu_global_size > 0) {
    #ifdef USE_GPU
    d_q = new sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
    #else
    d_q = new sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    #endif

    d_dataset = sycl::malloc_device<data_t>(size, *d_q);
    d_q->memcpy(d_dataset, dataset, sizeof(data_t) * size);
    d_result = sycl::malloc_device<sum_t>(wg_count, *d_q);

    sycl::range<1> gws (gpu_global_size);
    sycl::range<1> lws (wg_size);

    d_q->wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < params->repeat; i++) {
      d_q->submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<sum_t, 1> sm (sycl::range<1>(wg_size), cgh);
        cgh.parallel_for<class lr>(
         sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
         linear_regression(item, d_dataset, sm.get_pointer(), d_result);
        });
      });
    }
    d_q->wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    response->ktime += time;
  }

  if (cpu_offset > 0) {
    linear_regressionOMP(dataset, results, gpu_global_size, size);
  }

  if (gpu_global_size > 0) {
    d_q->memcpy(results, d_result, sizeof(sum_t) * wg_count * (100-cpu_offset) / 100).wait();
  }

  /* Finalize algorithm */
  sum_t final_result = {0.f, 0.f, 0.f, 0.f};

  for (size_t i = 0; i < wg_count; i++) {
    final_result.x() += results[i].x();
    final_result.y() += results[i].y();
    final_result.z() += results[i].z();
    final_result.w() += results[i].w();
  }

  double denom = (size * final_result.w() - (final_result.x() * final_result.x()));
  response->a0 = (final_result.y() * final_result.w() - final_result.x() * final_result.z()) / denom;
  response->a1 = (size * final_result.z() - final_result.x() * final_result.y()) / denom;
#ifdef DEBUG
  printf("%f %f %f\n", denom, response->a0, response->a1);
#endif

  /* Deallocate resources  */
  free(results);
  if (gpu_global_size > 0) {
    sycl::free(d_dataset, *d_q);
    sycl::free(d_result, *d_q);
  }

  r_squared(*d_q, params, dataset, &final_result, response);

  if (d_q != nullptr) delete(d_q);
}
