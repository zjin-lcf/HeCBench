#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <chrono>
#include <utility>  // std::swap
#include <hip/hip_runtime.h>
#include "utils.h"
#include "kernels.h"
#include "kernels_wrapper.h"

//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define HIP_CHECK(call)                                                    \
do {                                                                       \
    hipError_t err_ = call;                                                \
    if (err_ != hipSuccess) {                                              \
        fprintf(stderr, "HIP error at %s:%d code=%d(%s) \"%s\" \n",        \
                __FILE__, __LINE__, err_, hipGetErrorString(err_), #call); \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)


int main(int argc, char **argv) {
  if(argc < 3){
    printf("Usage: %s <file> <number of seams to remove> [options]\n"
           "valid options:\n-u\tupdate costs instead of recomputing them\n"
           "-a\tapproximate computation\n", argv[0]);
    return 1;
  }

  char *check;
  long seams_to_remove = strtol(argv[2], &check, 10);  //10 specifies base-10
  if (check == argv[2]){   //if no characters were converted pointers are equal
    printf("ERROR: can't convert string to number, exiting.\n");
    return 1;
  }

  int w, h, ncomp;
  unsigned char* imgv = stbi_load(argv[1], &w, &h, &ncomp, 0);
  if(imgv == NULL){
    printf("ERROR: can't load image \"%s\", exiting.\n", argv[1]);
    printf("Reason: %s\n", stbi_failure_reason());
    return 1;
  }

  if(ncomp != 3){
    printf("ERROR: image does not have 3 components (RGB), exiting.\n");
    return 1;
  }

  if(seams_to_remove < 0 || seams_to_remove >= w){
    printf("ERROR: number of seams to remove is invalid, exiting.\n");
    return 1;
  }

  seam_carver_mode mode = SEAM_CARVER_STANDARD_MODE;

  if(argc >= 4){
    if(strcmp(argv[3],"-u") == 0){
      mode = SEAM_CARVER_UPDATE_MODE;
      printf("update mode selected.\n");
    }
    else if(strcmp(argv[3],"-a") == 0){
      mode = SEAM_CARVER_APPROX_MODE;
      printf("approximation mode selected.\n");
    }
    else{
      printf("an invalid option was specified and will be ignored. Valid options are: -u, -a.\n");
    }
  }

  printf("Image loaded. Resizing...\n");

  int current_w = w;
  uchar4 *h_pixels = build_pixels(imgv, w, h);
  const size_t img_bytes = (size_t)w * h * sizeof(uchar4);
  const int cost_bytes = w * h * sizeof(short);
  const int h_bytes = h * sizeof(int);
  const int w_bytes = w * sizeof(int);

  uchar4 *d_pixels;
  uchar4 *d_pixels_swap;
  short *d_costs_left, *d_costs_swap_left;
  short *d_costs_up, *d_costs_swap_up;
  short *d_costs_right, *d_costs_swap_right;
  int *d_index_map;
  int *d_offset_map;
  int *d_indices_ref;
  int *d_indices;
  int *d_seam;
  int *reduce_row; //M row to consider for reduce
  int *d_M;

  if(mode != SEAM_CARVER_APPROX_MODE) {
    HIP_CHECK(hipMalloc((void**)&d_costs_left, cost_bytes));
    HIP_CHECK(hipMalloc((void**)&d_costs_up, cost_bytes));
    HIP_CHECK(hipMalloc((void**)&d_costs_right, cost_bytes));
  }
  if(mode == SEAM_CARVER_UPDATE_MODE){
    HIP_CHECK(hipMalloc((void**)&d_costs_swap_left, cost_bytes));
    HIP_CHECK(hipMalloc((void**)&d_costs_swap_up, cost_bytes));
    HIP_CHECK(hipMalloc((void**)&d_costs_swap_right, cost_bytes));
  }
  //sum map in approx mode
  HIP_CHECK(hipMalloc((void**)&d_M, img_bytes));

  // rows to consider for reduce
  if(mode == SEAM_CARVER_APPROX_MODE)
    reduce_row = d_M; //first row
  else
    reduce_row = d_M + w*(h-1); //last row

  if(mode == SEAM_CARVER_APPROX_MODE){
    HIP_CHECK(hipMalloc((void**)&d_index_map, img_bytes));
    HIP_CHECK(hipMalloc((void**)&d_offset_map, img_bytes));
  }

  int* indices = (int*)malloc(w_bytes);
  for(int i = 0; i < w; i++) indices[i] = i;

  HIP_CHECK(hipMalloc((void**)&d_indices, w_bytes));

  HIP_CHECK(hipMalloc((void**)&d_indices_ref, w_bytes));
  HIP_CHECK(hipMemcpy(d_indices_ref, indices, w_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc((void**)&d_seam, h_bytes));

  HIP_CHECK(hipMalloc((void**)&d_pixels, img_bytes));
  HIP_CHECK(hipMemcpy(d_pixels, h_pixels, img_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc((void**)&d_pixels_swap, img_bytes));

  if(mode == SEAM_CARVER_UPDATE_MODE)
    compute_costs(current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

  int num_iterations = 0;

  auto start = std::chrono::steady_clock::now();

  while(num_iterations < (int)seams_to_remove){

    if(mode == SEAM_CARVER_STANDARD_MODE)
      compute_costs(current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

    if(mode != SEAM_CARVER_APPROX_MODE){
      compute_M(current_w, w, h, d_M, d_costs_left, d_costs_up, d_costs_right);
      find_min_index(current_w, d_indices_ref, d_indices, reduce_row);
      find_seam(current_w, w, h, d_M, d_indices, d_seam);
    }
    else{
      approx_setup(current_w, w, h, d_pixels, d_index_map, d_offset_map, d_M);
      approx_M(current_w, w, h, d_offset_map,  d_M);
      find_min_index(current_w, d_indices_ref, d_indices, reduce_row);
      approx_seam(w, h, d_index_map, d_indices, d_seam);
    }

    remove_seam(current_w, w, h, d_M, d_pixels, d_pixels_swap, d_seam);
    std::swap(d_pixels, d_pixels_swap);

    if(mode == SEAM_CARVER_UPDATE_MODE){
      update_costs(current_w, w, h, d_M, d_pixels,
                   d_costs_left, d_costs_up, d_costs_right,
                   d_costs_swap_left, d_costs_swap_up, d_costs_swap_right, d_seam );
      std::swap(d_costs_left, d_costs_swap_left);
      std::swap(d_costs_up, d_costs_swap_up);
      std::swap(d_costs_right, d_costs_swap_right);
    }

    current_w--;
    num_iterations++;
  }

  HIP_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  float time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of seam carver kernels: %f (ms)\n", time * 1e-6f);

  HIP_CHECK(hipMemcpy(h_pixels, d_pixels, img_bytes, hipMemcpyDeviceToHost));
  unsigned char* output = flatten_pixels(h_pixels, w, h, current_w);
  printf("Image resized\n");

  printf("Saving in resized.bmp...\n");
  int success = stbi_write_bmp("resized.bmp", current_w, h, 3, output);
  printf("%s\n", success ? "Success" : "Failed");

  HIP_CHECK(hipFree(d_pixels));
  HIP_CHECK(hipFree(d_pixels_swap));
  if(mode != SEAM_CARVER_APPROX_MODE){
    HIP_CHECK(hipFree(d_costs_left));
    HIP_CHECK(hipFree(d_costs_up));
    HIP_CHECK(hipFree(d_costs_right));
  }
  if(mode == SEAM_CARVER_UPDATE_MODE){
    HIP_CHECK(hipFree(d_costs_swap_left));
    HIP_CHECK(hipFree(d_costs_swap_up));
    HIP_CHECK(hipFree(d_costs_swap_right));
  }
  HIP_CHECK(hipFree(d_M));
  HIP_CHECK(hipFree(d_indices));
  HIP_CHECK(hipFree(d_indices_ref));
  HIP_CHECK(hipFree(d_seam));
  if(mode == SEAM_CARVER_APPROX_MODE){
    HIP_CHECK(hipFree(d_index_map));
    HIP_CHECK(hipFree(d_offset_map));
  }
  free(h_pixels);
  free(output);
  free(indices);
  free(imgv);
  return 0;
}
