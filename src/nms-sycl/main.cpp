/*
 * NMS Benchmarking Framework
 *
 * "Work-Efficient Parallel Non-Maximum Suppression Kernels"
 * Copyright (c) 2019 David Oro et al.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32

using float4 = sycl::float4;

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n");
  printf("               repeat         -> Kernel execution count\n\n");
}

/* Gets the optimal X or Y dimension for a given thread block */
int get_optimal_dim(int val)
{
  int div, neg, cntneg, cntpos;

  /* We start figuring out if 'val' is divisible by 16
     (e.g. optimal 16x16 thread block of maximum GPU occupancy */

  neg = 1;
  div = 16;
  cntneg = div;
  cntpos = div;

  /* In order to guarantee the ending of this loop if 'val' is
     a prime number, we limit the loop to 5 iterations */

  for(int i=0; i<5; i++)
  {
    if(val % div == 0)
      return div;

    if(neg)
    {
      cntneg--;
      div = cntneg;
      neg = 0;
    }
    else
    {
      cntpos++;
      div = cntpos;
      neg = 1;
    }
  }

  return 16;
}

/* Gets an upper limit for 'val' multiple of the 'mul' integer */
int get_upper_limit(int val, int mul)
{
  int cnt = mul;

  /* The upper limit must be lower than
     the maximum allowed number of detections */

  while(cnt < val)
    cnt += mul;

  if(cnt > MAX_DETECTIONS)
    cnt = MAX_DETECTIONS;

  return cnt;
}

int main(int argc, char *argv[])
{
  int x, y, w;
  float score;

  if(argc != 4)
  {
    print_help();
    return 0;
  }

  /* Read input detection coordinates from the text file */
  int ndetections = 0;

  FILE *fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("Error: Unable to open file %s for input detection coordinates.\n", argv[1]);
    return -1;
  }

  /* Memory allocation in the host memory address space */
  size_t size = sizeof(float4) * MAX_DETECTIONS;
  float4* cpu_points = (float4*) malloc (size);
  if(!cpu_points)
  {
    printf("Error: Unable to allocate CPU memory.\n");
    return -1;
  }

  memset(cpu_points, 0, size);

  while(!feof(fp))
  {
    int cnt = fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);

    if (cnt !=4)
    {
       printf("Error: Invalid file format in line %d when reading %s\n", ndetections, argv[1]);
       return -1;
    }

    cpu_points[ndetections].x() = (float) x;       // x coordinate
    cpu_points[ndetections].y() = (float) y;       // y coordinate
    cpu_points[ndetections].z() = (float) w;       // window dimensions
    cpu_points[ndetections].w() = score;           // score

    ndetections++;
  }

  printf("Number of detections read from input file (%s): %d\n", argv[1], ndetections);

  fclose(fp);

  /* CPU array for storing the detection bitmap */
  size_t pts_bm_size = sizeof(unsigned char) * MAX_DETECTIONS;
  unsigned char* cpu_pointsbitmap;
  cpu_pointsbitmap = (unsigned char*) malloc(pts_bm_size);
  memset(cpu_pointsbitmap, 0, pts_bm_size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  /* GPU array for storing the coordinates, dimensions and score of each detected object */
  float4 *rects = sycl::malloc_device<float4>(MAX_DETECTIONS, q);
  q.memcpy(rects, cpu_points, size);

  /* GPU array for storing the non-maximum supression bitmap */
  size_t nms_bm_size = sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS;
  unsigned char *nmsbitmap = sycl::malloc_device<unsigned char>(MAX_DETECTIONS * MAX_DETECTIONS, q);
  q.memset(nmsbitmap, (unsigned char)1, nms_bm_size);

  /* GPU array for storing the detection bitmap */
  unsigned char *pointsbitmap = sycl::malloc_device<unsigned char>(MAX_DETECTIONS, q);
  q.memset(pointsbitmap, (unsigned char)0, pts_bm_size);

  /* Execute NMS on the GPU */

  /* We build up the non-maximum supression bitmap matrix by removing overlapping windows */
  int repeat = atoi(argv[3]);
  int limit = get_upper_limit(ndetections, 16);
  sycl::range<2> gen_gws(limit, limit);
  sycl::range<2> gen_lws(get_optimal_dim(limit), get_optimal_dim(limit));

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class generate>(sycl::nd_range<2>(gen_gws, gen_lws), [=] (sycl::nd_item<2> item) {
        const int i = item.get_global_id(1);
        const int j = item.get_global_id(0);
        if(rects[i].w() < rects[j].w())
        {
          float area = (rects[j].z() + 1.0f) * (rects[j].z() + 1.0f);
          float w = sycl::fmax(0.0f, sycl::fmin(rects[i].x() + rects[i].z(), rects[j].x() + rects[j].z()) -
                    sycl::fmax(rects[i].x(), rects[j].x()) + 1.0f);
          float h = sycl::fmax(0.0f, sycl::fmin(rects[i].y() + rects[i].z(), rects[j].y() + rects[j].z()) -
                    sycl::fmax(rects[i].y(), rects[j].y()) + 1.0f);
          nmsbitmap[i * MAX_DETECTIONS + j] = (((w * h) / area) < 0.3f) && (rects[j].z() != 0);
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (generate_nms_bitmap): %f (s)\n", (time * 1e-9f) / repeat);

  /* Then we perform a reduction for generating a point bitmap vector */
  sycl::range<1> reduce_gws (ndetections * MAX_DETECTIONS / N_PARTITIONS);
  sycl::range<1> reduce_lws (MAX_DETECTIONS / N_PARTITIONS);

  start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class reduce>(sycl::nd_range<1>(reduce_gws, reduce_lws), [=] (sycl::nd_item<1> item) {
        auto g = item.get_group();
        int bid = item.get_group(0);
        int lid = item.get_local_id(0);
        int idx = bid * MAX_DETECTIONS + lid;

        pointsbitmap[bid] = (item.barrier(sycl::access::fence_space::local_space),
                             sycl::all_of_group(g, nmsbitmap[idx]));

        for(int i=0; i<(N_PARTITIONS-1); i++)
        {
          idx += MAX_DETECTIONS / N_PARTITIONS;
          pointsbitmap[bid] = (item.barrier(sycl::access::fence_space::local_space),
                              sycl::all_of_group(g, pointsbitmap[bid] && nmsbitmap[idx]));
        }
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (reduce_nms_bitmap): %f (s)\n", (time * 1e-9f) / repeat);

  /* Dump detections after having performed the NMS */
  q.memcpy(cpu_pointsbitmap, pointsbitmap, pts_bm_size).wait();

  sycl::free(pointsbitmap, q);
  sycl::free(nmsbitmap, q);
  sycl::free(rects, q);

  fp = fopen(argv[2], "w");
  if (!fp)
  {
    printf("Error: Unable to open file %s for detection outcome.\n", argv[2]);
    return -1;
  }

  int totaldets = 0;
  for(int i = 0; i < ndetections; i++)
  {
    if(cpu_pointsbitmap[i])
    {
      x = (int) cpu_points[i].x();          // x coordinate
      y = (int) cpu_points[i].y();          // y coordinate
      w = (int) cpu_points[i].z();          // window dimensions
      score = cpu_points[i].w();            // score
      fprintf(fp, "%d,%d,%d,%f\n", x, y, w, score);
      totaldets++;
    }
  }
  fclose(fp);
  printf("Detections after NMS: %d\n", totaldets);

  free(cpu_points);
  free(cpu_pointsbitmap);

  return 0;
}
