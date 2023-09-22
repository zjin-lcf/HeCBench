/*
 * (c) 2009-2010 Christoph Schied <Christoph.Schied@uni-ulm.de>
 *
 * This file is part of flame.
 *
 * flame is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * flame is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with flame.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "iteration.cpp"

/* Each thread has a fixed assigned function and picks a random data-set by
 * indexing into the data array through a permutation array. The threads write
 * out their calculated data and read another randomly selected point by
 * indexing into another permutation array.
 *
 * This approach needs synchronization across all threads and threadblocks. To
 * achieve this, the algorithm has been split into three kernels:
 *  - initialize: initializes the iterations by selecting a random point of
 *    the set of starting points and applies one iteration
 *  - iterate: reads one point, iterates, writes out
 *  - generate points: reads iterated points and generates a bigger amount of
 *    points out of those points.
 *
 */

/* calculates the function a thread has to evaluate by using a binary search
 */
int
get_function_idx(int idx, const ConstMemParams &params)
{
  int func;
  int start = 0, end = NUM_FUNCTIONS;

  /* use warp size granularity, decreases branch divergence */
  idx &= ~31;

  for(int i = 0; i < 6; i++) {
    func = (start + end) / 2;
    if(params.thread_function_mapping[func] <= idx)
      start = func + 1;
    else
      end = func;
  }

  return func;
}

/* selects a random point out of the set of startingpoints and calculates one
 * iteration
 */
void
kernel_initialize(sycl::short2 *short_points, 
                  short *colors, 
                  const unsigned short *perms, 
                  const int perm_num,
                  sycl::float2 *start_pos,
                  const float *random_numbers,
                  const ConstMemParams params,
                  sycl::nd_item<1> &item)
{
  int idx = item.get_global_id(0);

  int function = get_function_idx(idx, params);


  int perm_idx = perms[NUM_THREADS * perm_num + idx];
  sycl::float2 point = start_pos[perm_idx];

  float color = 0.5f;

  iteration_fractal_flame(point, color, function, idx, random_numbers, params);

  colors[idx] = sycl::vec<float, 1>{color}
                    .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  short_points[idx].x() =
      sycl::vec<float, 1>{point.x()}
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  short_points[idx].y() =
      sycl::vec<float, 1>{point.y()}
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

/* reads one point, applies one iteration step and writes the point out */
void
kernel_iterate(sycl::short2 *short_points, 
               short *colors, 
               const unsigned short *perms,
               const int perm_num,
               const float *random_numbers,
               const ConstMemParams params,
               sycl::nd_item<1> &item)
{
  int idx = item.get_global_id(0);

  int function = get_function_idx(idx, params);

  sycl::float2 point;
  float color;

  int perm_idx = perms[NUM_THREADS * perm_num + idx];

  color = sycl::vec<sycl::half, 1>{colors[perm_idx]}
              .convert<float, sycl::rounding_mode::automatic>()[0];
  point.x() = sycl::vec<sycl::half, 1>{short_points[perm_idx].x()}
                  .convert<float, sycl::rounding_mode::automatic>()[0];
  point.y() = sycl::vec<sycl::half, 1>{short_points[perm_idx].y()}
                  .convert<float, sycl::rounding_mode::automatic>()[0];

  iteration_fractal_flame(point, color, function, idx, random_numbers, params);

  colors[perm_idx] = sycl::vec<float, 1>{color}
                         .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  short_points[perm_idx].x() =
      sycl::vec<float, 1>{point.x()}
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  short_points[perm_idx].y() =
      sycl::vec<float, 1>{point.y()}
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

/* after applying n iterations, the points converged close enough. new points
 * are created out of these converged points by applying another iteration
 * step and these new points get written into the destination VBO
 */
void
kernel_generate_points(sycl::float3 *vertices,
                       sycl::short2 *short_points, 
                       short *colors, 
                       const unsigned short *perms, 
                       const int perm_num,
                       const float *random_numbers,
                       const ConstMemParams params,
                       sycl::nd_item<1> &item)
{
  int idx = item.get_global_id(0);

  int function = get_function_idx(idx, params);

  sycl::float2 point;
  float color;

  for(int i = 0; i < NUM_POINTS_PER_THREAD; i++) {
    int perm_idx = perms[((perm_num + i) % NUM_PERMUTATIONS) *
      NUM_THREADS + idx];

    sycl::short2 _p = short_points[perm_idx];
    point.x() = sycl::vec<sycl::half, 1>{_p.x()}
                    .convert<float, sycl::rounding_mode::automatic>()[0];
    point.y() = sycl::vec<sycl::half, 1>{_p.y()}
                    .convert<float, sycl::rounding_mode::automatic>()[0];
    color = sycl::vec<sycl::half, 1>{colors[perm_idx]}
                .convert<float, sycl::rounding_mode::automatic>()[0];

    iteration_fractal_flame(point, color, function, idx, random_numbers, params);

    vertices[idx + i * NUM_THREADS].x() = point.x();
    vertices[idx + i * NUM_THREADS].y() = point.y();
    vertices[idx + i * NUM_THREADS].z() = color;
  }
}

