#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
  int size_image = i_img_width*i_img_height;
  size_t mem_size_image = sizeof(DTYPE) * size_image;
  DTYPE *h_image  = (DTYPE*)malloc(mem_size_image * i_img_count);

  for(int j=0;j<i_img_count;j++)
  {
    for(int i=0;i<size_image;i++)
    {
      h_image[(j*size_image)+i] = rand()%256 / (DTYPE)255;
    }
  }

  // host and device results
  int size_output = o_img_width * o_img_height;
  size_t mem_size_output = sizeof(DTYPE) * size_output;
  DTYPE* h_output = (DTYPE*) malloc(mem_size_output*i_img_count);
  DTYPE* d_output = (DTYPE*) malloc(mem_size_output*i_img_count);

  // filter size same as stride size
  const int pool_width  = Hstride;
  const int pool_height = Vstride;

#pragma omp target data map(to: h_image[0:size_image*i_img_count]) \
                        map(from: d_output[0:size_output*i_img_count])
{
  for (int n = 0; n < 100; n++) {
    #pragma omp target teams distribute parallel for collapse(3) thread_limit(256) 
    for (int z = 0; z < i_img_count; z++) {
      for (int y = 0; y < o_img_height; y++) {
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
          d_output[(((z*o_img_height)+y)*o_img_width)+x] = maxval;
        }
      }
    }
  }
} 

  // verification using the CPU results
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
            maxval = fmaxf(maxval, h_image[idxIn]);
          }
        }
        h_output[(((z * o_img_height) + y) * o_img_width) + x] = maxval;
      }

  int status = memcmp(h_output, d_output, sizeof(DTYPE)*i_img_count*o_img_height*o_img_width);
  if (status == 0)
    printf("PASS\n");
  else
    printf("FAIL\n");

  free(h_image);
  free(h_output);
  free(d_output);
  return status;
}
