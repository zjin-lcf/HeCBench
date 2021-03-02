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
#include "common.h"

#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n\n");
}


/* Gets the optimal X or Y dimension for a given CUDA block */
int get_optimal_dim(int val)
{
  int div, neg, cntneg, cntpos;


  /* We start figuring out if 'val' is divisible by 16 
     (e.g. optimal 16x16 CUDA block of maximum GPU occupancy */

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

  if(argc != 3)
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
  float4* cpu_points = (float4*) malloc(sizeof(float4) * MAX_DETECTIONS);
  if(!cpu_points)
  {
    printf("Error: Unable to allocate CPU memory.\n");
    return -1;
  }

  memset(cpu_points, 0, sizeof(float4) * MAX_DETECTIONS);

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
  unsigned char* cpu_pointsbitmap;
  cpu_pointsbitmap = (unsigned char*) malloc(sizeof(unsigned char) * MAX_DETECTIONS);
  memset(cpu_pointsbitmap, 0, sizeof(unsigned char) * MAX_DETECTIONS);


#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  /* GPU array for storing the coordinates, dimensions and score of each detected object */
  buffer<float4, 1> d_points(cpu_points, MAX_DETECTIONS);

  /* GPU array for storing the non-maximum supression bitmap */
  buffer<unsigned char, 1> d_nmsbitmap (MAX_DETECTIONS * MAX_DETECTIONS);

  /* GPU array for storing the detection bitmap */
  buffer<unsigned char, 1> d_pointsbitmap (MAX_DETECTIONS);

  /* Execute NMS on the GPU */
  q.submit([&] (handler &cgh) {
    auto nmsbitmap = d_nmsbitmap.get_access<sycl_write>(cgh);
    cgh.fill(nmsbitmap, (unsigned char)1);
  });

  q.submit([&] (handler &cgh) {
    auto pointsbitmap = d_pointsbitmap.get_access<sycl_write>(cgh);
    cgh.fill(pointsbitmap, (unsigned char)0);
  });


  /* We build up the non-maximum supression bitmap matrix by removing overlapping windows */
  // generate_nms_bitmap<<<pkgrid, pkthreads>>>(points, nmsbitmap, 0.3f);
  int limit = get_upper_limit(ndetections, 16);
  range<2> gen_gws(limit, limit);
  range<2> gen_lws(get_optimal_dim(limit), get_optimal_dim(limit));

  for (int n = 0; n < 100; n++) {
    q.submit([&] (handler &cgh) {
      auto rects = d_points.get_access<sycl_read>(cgh);
      auto nmsbitmap = d_nmsbitmap.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class generate_nms_bitmap>(nd_range<2>(gen_gws, gen_lws), [=] (nd_item<2> item) {
        const int i = item.get_global_id(1);
        const int j = item.get_global_id(0);
        if(rects[i].w() < rects[j].w())
        {
          float area = (rects[j].z() + 1.0f) * (rects[j].z() + 1.0f);
          float w = cl::sycl::max(0.0f, cl::sycl::min(rects[i].x() + rects[i].z(), rects[j].x() + rects[j].z()) - 
                    cl::sycl::max(rects[i].x(), rects[j].x()) + 1.0f);
          float h = cl::sycl::max(0.0f, cl::sycl::min(rects[i].y() + rects[i].z(), rects[j].y() + rects[j].z()) - 
                    cl::sycl::max(rects[i].y(), rects[j].y()) + 1.0f);
          nmsbitmap[i * MAX_DETECTIONS + j] = (((w * h) / area) < 0.3f) && (rects[j].z() != 0);
        } 
      });
    });
  }

 
  /* Then we perform a reduction for generating a point bitmap vector */
  // reduce_nms_bitmap<<<pkgrid, pkthreads>>>(nmsbitmap, pointsbitmap, ndetections);
  range<1> reduce_gws (ndetections * MAX_DETECTIONS / N_PARTITIONS); 
  range<1> reduce_lws (MAX_DETECTIONS / N_PARTITIONS); 

  for (int n = 0; n < 100; n++) {
    q.submit([&] (handler &cgh) {
      auto nmsbitmap= d_nmsbitmap.get_access<sycl_read>(cgh);
      auto pointsbitmap= d_pointsbitmap.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class reduce_nms_bitmap>(nd_range<1>(reduce_gws, reduce_lws), [=] (nd_item<1> item) {
        auto g = item.get_group();
        int bid = item.get_group(0);
        int lid = item.get_local_id(0);
        int idx = bid * MAX_DETECTIONS + lid;

        pointsbitmap[bid] = ( item.barrier(access::fence_space::local_space),
              ONEAPI::all_of(g, nmsbitmap[idx]));

        for(int i=0; i<(N_PARTITIONS-1); i++)
        {
          idx += MAX_DETECTIONS / N_PARTITIONS;
          pointsbitmap[bid] = ( item.barrier(access::fence_space::local_space),
              ONEAPI::all_of(g, pointsbitmap[bid] && nmsbitmap[idx]));
        }
      });
    });
  }

  /* Dump detections after having performed the NMS */
  q.submit([&] (handler &cgh) {
    auto pointsbitmap = d_pointsbitmap.get_access<sycl_write>(cgh);
    cgh.copy(pointsbitmap, cpu_pointsbitmap);
  });
  q.wait();

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

