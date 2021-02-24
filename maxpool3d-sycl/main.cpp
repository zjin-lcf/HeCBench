#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "common.h"

typedef float DTYPE;

int main(int argc, char** argv)
{
  srand(2);
  int Hstride=2, Vstride=2;
  int i_img_width  = atoi(argv[1]);  
  int i_img_height = atoi(argv[2]);
  int i_img_count = atoi(argv[3]);
  int o_img_width  = i_img_width/Hstride;
  int o_img_height = i_img_height/Vstride;

  printf("input image width %d Hstride %d\n", i_img_width,Hstride);
  printf("input image height %d Vstride %d\n", i_img_height,Vstride);
  printf("output image width %d\n", o_img_width);
  printf("output image height %d\n", o_img_height);

  // Generate random values for each image
  unsigned int size_image = i_img_width * i_img_height;
  unsigned int mem_size_image = sizeof(DTYPE) * size_image;
  DTYPE *h_image  = (DTYPE*)malloc(mem_size_image * i_img_count);

  for(int j=0;j<i_img_count;j++)
  {
    for(int i=0;i<size_image;i++)
    {
      h_image[(j*size_image)+i] = rand()%256 / (DTYPE)255;
    }
  }

  unsigned int size_output = o_img_width * o_img_height;
  unsigned int mem_size_output = sizeof(DTYPE) * size_output;
  // host result
  DTYPE* h_output = (DTYPE*) malloc(mem_size_output*i_img_count);
  // device result 
  DTYPE* d_output = (DTYPE*) malloc(mem_size_output*i_img_count);

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);

  // Create the input and output arrays in device memory 
  buffer<DTYPE, 1> d_image(h_image, size_image*i_img_count);
  buffer<DTYPE, 1> d_result(d_output, size_output*i_img_count);

  // assume output image dimensions are multiple of 16
  range<3> localWorkSize(1, 16, 16);
  range<3> globalWorkSize(i_img_count, o_img_height, o_img_width);

  // filter size same as stride size
  const int pool_width  = Hstride;
  const int pool_height = Vstride;

  for (int n = 0; n < 100; n++) {
    q.submit([&] (handler &h) {
      auto i_img = d_image.get_access<sycl_read>(h);
      auto o_img = d_result.get_access<sycl_discard_write>(h);
      h.parallel_for<class maxpool3>(nd_range<3>(globalWorkSize, localWorkSize), [=] (nd_item<3> item) {
        const int x = item.get_global_id(2); 
        const int y = item.get_global_id(1);
        const int z = item.get_global_id(0);
        const int xidx = Hstride*x;
        const int yidx = Vstride*y;
        DTYPE maxval = (DTYPE)0;

        for (int r = 0; r < pool_height; r++) 
        { 
          const int idxIntmp = ((z*i_img_height + yidx + r) * i_img_width) + xidx;
          for(int c = 0; c < pool_width; c++)
          {
            const int idxIn = idxIntmp + c;
            maxval = cl::sycl::fmax(maxval,i_img[idxIn]);
          }
        }
        o_img[(((z * o_img_height) + y) * o_img_width) + x] = maxval;
      });
    });
  }

  } // sycl scope

  // verification using the CPU results
  const int pool_width  = Hstride;
  const int pool_height = Vstride;
  for (int z = 0; z < i_img_count; z++)
    for (int y = 0; y < o_img_height; y++)
      for (int x = 0; x < o_img_width; x++) {
        const int xidx = Hstride*x;
        const int yidx = Vstride*y;
        DTYPE maxval = (DTYPE)0;
        for (int r = 0; r < pool_height; r++) 
        { 
          const int idxIntmp = ((z*i_img_height + yidx + r) * i_img_width) + xidx;
          for(int c = 0; c < pool_width; c++)
          {
            const int idxIn = idxIntmp + c;
            maxval = fmaxf(maxval,h_image[idxIn]);
          }
        }
        h_output[(((z*o_img_height)+y)*o_img_width)+x] = maxval;
      }

  int status = memcmp(h_output, d_output, sizeof(DTYPE)*i_img_count*o_img_width*o_img_height);
  if (status == 0)
    printf("PASS\n");
  else
    printf("FAIL\n");

  free(h_image);
  free(h_output);
  free(d_output);
  return status;
}
