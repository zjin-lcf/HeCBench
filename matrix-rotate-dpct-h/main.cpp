#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdio>

void rotate_matrix_parallel (float *matrix, const int n,
                             sycl::nd_item<3> item_ct1) {
  int layer = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
              item_ct1.get_local_id(2);
  if (layer < n/2) {
    int first = layer;
    int last = n - 1 - layer;
    for(int i = first; i < last; ++i) {
      int offset = i - first;

      float top = matrix[first*n+i]; // save top
      // left -> top
      matrix[first*n+i] = matrix[(last-offset)*n+first];

      // bottom -> left
      matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];

      // right -> bottom
      matrix[last*n+(last-offset)] = matrix[i*n+last];

      // top -> right
      matrix[i*n+last] = top; // right <- saved top
    }
  }
}

void rotate_matrix_serial(float *matrix, int n) {

  for (int layer = 0; layer < n / 2; ++layer) {
    int first = layer;
    int last = n - 1 - layer;
    for(int i = first; i < last; ++i) {
      int offset = i - first;
        float top = matrix[first*n+i]; // save top
        // left -> top
        matrix[first*n+i] = matrix[(last-offset)*n+first];

        // bottom -> left
        matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];

        // right -> bottom
        matrix[last*n+(last-offset)] = matrix[i*n+last];

        // top -> right
        matrix[i*n+last] = top; // right <- saved top
    }
  }
}

int main(int argc, char** argv) {

  const int n = atoi(argv[1]);
  float *serial_res = (float*) aligned_alloc(1024, n*n*sizeof(float));
  float *parallel_res = (float*) aligned_alloc(1024, n*n*sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      serial_res[i*n+j] = parallel_res[i*n+j] = i*n+j;

  float *d_parallel_res;
  dpct::dpct_malloc((void **)&d_parallel_res, n * n * sizeof(float));
  dpct::dpct_memcpy(d_parallel_res, parallel_res, n * n * sizeof(float),
                    dpct::host_to_device);

  for (int i = 0; i < 100; i++) {
    rotate_matrix_serial(serial_res, n);
    {
      dpct::buffer_t d_parallel_res_buf_ct0 = dpct::get_buffer(d_parallel_res);
      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_parallel_res_acc_ct0 =
            d_parallel_res_buf_ct0.get_access<sycl::access::mode::read_write>(
                cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (n / 2 + 255) / 256) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
              rotate_matrix_parallel((float *)(&d_parallel_res_acc_ct0[0]), n,
                                     item_ct1);
            });
      });
    }
  }
  dpct::dpct_memcpy(parallel_res, d_parallel_res, n * n * sizeof(float),
                    dpct::device_to_host);

  int errors = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (serial_res[i*n+j] != parallel_res[i*n+j]) {
        errors++; 
        break;
      }
    }
  }
  if (errors) 
    printf("fail\n");
  else 
    printf("success\n");

  free(serial_res);
  free(parallel_res);
  dpct::dpct_free(d_parallel_res);
  return 0;
}

