#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef float DTYPE;

void
maxpool3d(const DTYPE* i_img, DTYPE* o_img, 
      const int Hstride,
      const int Vstride,
      const int pool_width,
      const int pool_height,
      const int i_img_width,
      const int i_img_height,
      const int o_img_width,
      const int o_img_height, sycl::nd_item<3> item_ct1 )
{

  const int x = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                item_ct1.get_local_id(2);
  const int y = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) +
                item_ct1.get_local_id(1);
  const int z = item_ct1.get_local_range().get(0) * item_ct1.get_group(0) +
                item_ct1.get_local_id(0);
  const int xidx = Hstride*x;
  const int yidx = Vstride*y;
  DTYPE maxval = (DTYPE)0;

  for (int r = 0; r < pool_height; r++) 
  { 
    const int idxIntmp = ((z*i_img_height + yidx + r) * i_img_width) + xidx;
    for(int c = 0; c < pool_width; c++)
    {
      const int idxIn = idxIntmp + c;
      maxval = sycl::fmax((float)maxval, (float)(i_img[idxIn]));
    }
  }
  o_img[(((z*o_img_height)+y)*o_img_width)+x] = maxval;
}

int main(int argc, char** argv)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  int mem_size_image = sizeof(DTYPE) * size_image;
  DTYPE *h_image  = (DTYPE*)malloc(mem_size_image * i_img_count);

  for(int j=0;j<i_img_count;j++)
  {
    for(int i=0;i<size_image;i++)
    {
      h_image[(j*size_image)+i] = rand()%256 / (DTYPE)255;
    }
  }

  int size_output = o_img_width * o_img_height;
  int mem_size_output = sizeof(DTYPE) * size_output;
  // host result
  DTYPE* h_output = (DTYPE*) malloc(mem_size_output*i_img_count);
  // device result 
  DTYPE* d_output = (DTYPE*) malloc(mem_size_output*i_img_count);

  // Create the input and output arrays in device memory 
  DTYPE* d_image;
  d_image = (DTYPE *)sycl::malloc_device(mem_size_image * i_img_count, q_ct1);
  q_ct1.memcpy(d_image, h_image, mem_size_image * i_img_count).wait();

  DTYPE* d_result;
  d_result = (DTYPE *)sycl::malloc_device(mem_size_output * i_img_count, q_ct1);

  // assume output image dimensions are multiple of 16
  sycl::range<3> block_dim(16, 16, 1);
  sycl::range<3> grid_dim(o_img_width / 16, o_img_height / 16, i_img_count);

  // filter size same as stride size
  const int pool_width  = Hstride;
  const int pool_height = Vstride;

  for (int n = 0; n < 100; n++) {
    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = grid_dim * block_dim;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(block_dim.get(2), block_dim.get(1),
                                           block_dim.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            maxpool3d(d_image, d_result, Hstride, Vstride, pool_width,
                      pool_height, i_img_width, i_img_height, o_img_width,
                      o_img_height, item_ct1);
          });
    });
  }
  q_ct1.memcpy(d_output, d_result, mem_size_output * i_img_count).wait();

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
  sycl::free(d_image, q_ct1);
  sycl::free(d_result, q_ct1);
  return status;
}
