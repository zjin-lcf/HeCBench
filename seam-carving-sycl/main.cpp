#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <utility>  // std::swap
#include "common.h"
#include "utils.h"
#include "kernels.h"
#include "kernels_wrapper.h"

//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char **argv) {
  if(argc < 3){
    printf("usage: %s <file> <number of seams to remove> [options]\n"
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
    printf("ERROR: can't load image \"%s\" (maybe the file does not exist?), exiting.\n", argv[1]);
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
  const int img_size = w * h;
  const int w_bytes = w * sizeof(int);

  int* indices = (int*)malloc(w_bytes);
  for(int i = 0; i < w; i++) indices[i] = i;

  { // sycl scope
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // remove the conditions when buffers are instantiated
  buffer<short, 1> d_costs_left (img_size);
  buffer<short, 1> d_costs_up (img_size);
  buffer<short, 1> d_costs_right (img_size);
  buffer<short, 1> d_costs_swap_left (img_size);
  buffer<short, 1> d_costs_swap_up (img_size);
  buffer<short, 1> d_costs_swap_right (img_size);
  buffer<int, 1> d_index_map (img_size);
  buffer<int, 1> d_offset_map (img_size);

  //sum map in approx mode
  buffer<int, 1> d_M (img_size);

  // rows to consider for reduce
  id<1> index = (mode == SEAM_CARVER_APPROX_MODE) ? 0 : w*(h-1);
  range<1> subRange = (mode == SEAM_CARVER_APPROX_MODE) ? img_size : img_size - w*(h-1);
  buffer<int, 1> reduce_row (d_M, index, subRange);

  buffer<int, 1> d_indices (w);
  buffer<int, 1> d_indices_ref (indices, w);
  buffer<int, 1> d_seam (h);

  buffer<uchar4, 1> d_pixels (h_pixels, img_size);
  buffer<uchar4, 1> d_pixels_swap (img_size);

  if(mode == SEAM_CARVER_UPDATE_MODE)
    compute_costs(q, current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

  int num_iterations = 0;
  while(num_iterations < (int)seams_to_remove){

    if(mode == SEAM_CARVER_STANDARD_MODE)
      compute_costs(q, current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

    if(mode != SEAM_CARVER_APPROX_MODE){
      compute_M(q, current_w, w, h, d_M, d_costs_left, d_costs_up, d_costs_right);
      find_min_index(q, current_w, d_indices_ref, d_indices, reduce_row);
      find_seam(q, current_w, w, h, d_M, d_indices, d_seam);
    }
    else{
      approx_setup(q, current_w, w, h, d_pixels, d_index_map, d_offset_map, d_M);
      approx_M(q, current_w, w, h, d_offset_map,  d_M);
      find_min_index(q, current_w, d_indices_ref, d_indices, reduce_row);
      approx_seam(q, w, h, d_index_map, d_indices, d_seam);
    }

    remove_seam(q, current_w, w, h, d_M, d_pixels, d_pixels_swap, d_seam);
    std::swap(d_pixels, d_pixels_swap);

    if(mode == SEAM_CARVER_UPDATE_MODE){
      update_costs(q, current_w, w, h, d_M, d_pixels, 
                   d_costs_left, d_costs_up, d_costs_right,
                   d_costs_swap_left, d_costs_swap_up, d_costs_swap_right, d_seam );
      std::swap(d_costs_left, d_costs_swap_left);
      std::swap(d_costs_up, d_costs_swap_up);
      std::swap(d_costs_right, d_costs_swap_right);
    }

    current_w--;
    num_iterations++;
  }
  }  // sycl scope
  unsigned char* output = flatten_pixels(h_pixels, w, h, current_w); 
  printf("Image resized\n");

  printf("Saving in resized.bmp...\n");
  int success = stbi_write_bmp("resized.bmp", current_w, h, 3, output);
  printf("%s\n", success ? "Success" : "Failed");

  free(h_pixels);
  free(output);   
  free(indices);
  return 0;
}
