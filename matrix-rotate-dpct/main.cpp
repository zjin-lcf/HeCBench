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

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const int n = atoi(argv[1]);
  float *serial_res = (float*) aligned_alloc(1024, n*n*sizeof(float));
  float *parallel_res = (float*) aligned_alloc(1024, n*n*sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      serial_res[i*n+j] = parallel_res[i*n+j] = i*n+j;

  float *d_parallel_res;
  d_parallel_res = sycl::malloc_device<float>(n * n, q_ct1);
  q_ct1.memcpy(d_parallel_res, parallel_res, n * n * sizeof(float)).wait();

  for (int i = 0; i < 100; i++) {
    rotate_matrix_serial(serial_res, n);
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, (n / 2 + 255) / 256) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
          [=](sycl::nd_item<3> item_ct1) {
            rotate_matrix_parallel(d_parallel_res, n, item_ct1);
          });
    });
  }
  q_ct1.memcpy(parallel_res, d_parallel_res, n * n * sizeof(float)).wait();

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
  sycl::free(d_parallel_res, q_ct1);
  return 0;
}

