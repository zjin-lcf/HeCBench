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

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>

#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n\n");
}

/* NMS Map kernel */
void generate_nms_bitmap(const sycl::float4 *rects, unsigned char *nmsbitmap,
                         const float othreshold, sycl::nd_item<3> item_ct1)
{
  const int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);
  const int j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                item_ct1.get_local_id(1);

  if (rects[i].w() < rects[j].w())
  {
    float area = (rects[j].z() + 1.0f) * (rects[j].z() + 1.0f);
    float w = sycl::max(
        0.0f, (float)(sycl::min((float)(rects[i].x() + rects[i].z()),
                                (float)(rects[j].x() + rects[j].z())) -
                      sycl::max(rects[i].x(), rects[j].x()) + 1.0f));
    float h = sycl::max(
        0.0f, (float)(sycl::min((float)(rects[i].y() + rects[i].z()),
                                (float)(rects[j].y() + rects[j].z())) -
                      sycl::max(rects[i].y(), rects[j].y()) + 1.0f));
    nmsbitmap[i * MAX_DETECTIONS + j] =
        (((w * h) / area) < othreshold) && (rects[j].z() != 0);
  } 
}


/* NMS Reduce kernel */
__inline__ void compute_nms_point_mask(unsigned char* pointsbitmap, int cond, int idx, int ndetections,
                                       sycl::nd_item<3> item_ct1)
{
  *pointsbitmap =
      (item_ct1.barrier(), sycl::ONEAPI::all_of(item_ct1.get_group(), cond));
}


void reduce_nms_bitmap(unsigned char* nmsbitmap, unsigned char* pointsbitmap, int ndetections,
                       sycl::nd_item<3> item_ct1)
{
  int idx = item_ct1.get_group(2) * MAX_DETECTIONS + item_ct1.get_local_id(2);

  compute_nms_point_mask(&pointsbitmap[item_ct1.get_group(2)], nmsbitmap[idx],
                         idx, ndetections, item_ct1);

  for(int i=0; i<(N_PARTITIONS-1); i++)
  {
    idx += MAX_DETECTIONS / N_PARTITIONS;
    compute_nms_point_mask(&pointsbitmap[item_ct1.get_group(2)],
                           pointsbitmap[item_ct1.get_group(2)] &&
                               nmsbitmap[idx],
                           idx, ndetections, item_ct1);
  }
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

int main(int argc, char *argv[]) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int res;
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
  sycl::float4 *cpu_points =
      (sycl::float4 *)malloc(sizeof(sycl::float4) * MAX_DETECTIONS);
  if(!cpu_points)
  {
    printf("Error: Unable to allocate CPU memory.\n");
    return -1;
  }

  memset(cpu_points, 0, sizeof(sycl::float4) * MAX_DETECTIONS);

  while(!feof(fp))
  {
     int cnt = fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);

     if (cnt !=4)
     {
	printf("Error: Invalid file format in line %d when reading %s\n", ndetections, argv[1]);
        return -1;
     }

    cpu_points[ndetections].x() = (float)x; // x coordinate
    cpu_points[ndetections].y() = (float)y; // y coordinate
    cpu_points[ndetections].z() = (float)w; // window dimensions
    cpu_points[ndetections].w() = score;    // score

    ndetections++;
  }

  printf("Number of detections read from input file (%s): %d\n", argv[1], ndetections);

  fclose(fp);

  /* CPU array for storing the detection bitmap */
  unsigned char* cpu_pointsbitmap;
  cpu_pointsbitmap = (unsigned char*) malloc(sizeof(unsigned char) * MAX_DETECTIONS);
  memset(cpu_pointsbitmap, 0, sizeof(unsigned char) * MAX_DETECTIONS);

  /* GPU array for storing the coordinates, dimensions and score of each detected object */

  int err;

  sycl::float4 *points;
  /*
  DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (points = (sycl::float4 *)dpct::dpct_malloc(sizeof(sycl::float4) *
                                                    MAX_DETECTIONS),
         0);

  /*
  DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err =
      (dpct::dpct_memset(points, 0, sizeof(sycl::float4) * MAX_DETECTIONS), 0);

  /* GPU array for storing the non-maximum supression bitmap */
  unsigned char* nmsbitmap;
  /*
  DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (nmsbitmap = (unsigned char *)dpct::dpct_malloc(
             sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS),
         0);

  /* GPU array for storing the detection bitmap */
  unsigned char* pointsbitmap;
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (pointsbitmap = (unsigned char *)dpct::dpct_malloc(
             sizeof(unsigned char) * MAX_DETECTIONS),
         0);

  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (dpct::dpct_memset(nmsbitmap, 1,
                           sizeof(unsigned char) * MAX_DETECTIONS *
                               MAX_DETECTIONS),
         0);

  /*
  DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (dpct::dpct_memset(pointsbitmap, 0,
                           sizeof(unsigned char) * MAX_DETECTIONS),
         0);

  /* Transfer detection coordinates read from the input text file to the GPU */
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (dpct::dpct_memcpy(points, cpu_points,
                           sizeof(sycl::float4) * MAX_DETECTIONS,
                           dpct::host_to_device),
         0);

  /* Execute NMS on the GPU */

  int limit = get_upper_limit(ndetections, 16);
  int pkthreads_x = get_optimal_dim(limit);
  int pkthreads_y = get_optimal_dim(limit);
  int pkgrid_x = limit / pkthreads_x;
  int pkgrid_y = limit / pkthreads_y;

  sycl::range<3> pkthreads(1, pkthreads_y, pkthreads_x);
  sycl::range<3> pkgrid(1, pkgrid_y, pkgrid_x);

  /* We build up the non-maximum supression bitmap matrix by removing overlapping windows */
  for (int n = 0; n < 100; n++)
    /*
    DPCT1049:14: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
  {
    dpct::buffer_t points_buf_ct0 = dpct::get_buffer(points);
    dpct::buffer_t nmsbitmap_buf_ct1 = dpct::get_buffer(nmsbitmap);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto points_acc_ct0 =
          points_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto nmsbitmap_acc_ct1 =
          nmsbitmap_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(pkgrid * pkthreads, pkthreads),
                       [=](sycl::nd_item<3> item_ct1) {
                         generate_nms_bitmap(
                             (const sycl::float4 *)(&points_acc_ct0[0]),
                             (unsigned char *)(&nmsbitmap_acc_ct1[0]), 0.3f,
                             item_ct1);
                       });
    });
  }

  /*
  DPCT1010:15: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  err = 0;

  pkthreads[2] = MAX_DETECTIONS / N_PARTITIONS;
  pkthreads[1] = 1;
  pkgrid[2] = ndetections;
  pkgrid[1] = 1;

  /* Then we perform a reduction for generating a point bitmap vector */
  for (int n = 0; n < 100; n++)
    /*
    DPCT1049:17: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
  {
    dpct::buffer_t nmsbitmap_buf_ct0 = dpct::get_buffer(nmsbitmap);
    dpct::buffer_t pointsbitmap_buf_ct1 = dpct::get_buffer(pointsbitmap);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto nmsbitmap_acc_ct0 =
          nmsbitmap_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto pointsbitmap_acc_ct1 =
          pointsbitmap_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(pkgrid * pkthreads, pkthreads),
                       [=](sycl::nd_item<3> item_ct1) {
                         reduce_nms_bitmap(
                             (unsigned char *)(&nmsbitmap_acc_ct0[0]),
                             (unsigned char *)(&pointsbitmap_acc_ct1[0]),
                             ndetections, item_ct1);
                       });
    });
  }

  /*
  DPCT1010:18: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  err = 0;

  /* Dump detections after having performed the NMS */

  /*
  DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (dpct::dpct_memcpy(cpu_pointsbitmap, pointsbitmap,
                           sizeof(unsigned char) * MAX_DETECTIONS,
                           dpct::device_to_host),
         0);

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
      x = (int)cpu_points[i].x(); // x coordinate
      y = (int)cpu_points[i].y(); // y coordinate
      w = (int)cpu_points[i].z(); // window dimensions
      score = cpu_points[i].w();  // score
      fprintf(fp, "%d,%d,%d,%f\n", x, y, w, score);
      totaldets++; 
    }
  }
  fclose(fp);
  printf("Detections after NMS: %d\n", totaldets);

  dpct::dpct_free(points);
  dpct::dpct_free(nmsbitmap);
  dpct::dpct_free(pointsbitmap);
  free(cpu_points);
  free(cpu_pointsbitmap);

  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
