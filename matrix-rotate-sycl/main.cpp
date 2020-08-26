#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "common.h"

void rotate_matrix_serial(float *matrix, const int n) {
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

  {
#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float,1> d_parallel_res(parallel_res, n*n);

  for (int i = 0; i < 100; i++) {
    rotate_matrix_serial(serial_res, n);
    q.submit([&](handler &h) {
      auto matrix = d_parallel_res.get_access<sycl_read_write>(h);
      h.parallel_for(nd_range<1>(range<1>((n/2+255)/256*256), range<1>(256)), [=](nd_item<1> item) {
        int layer = item.get_global_id(0); 
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
      });
    });
  }
  q.wait();
  }

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
  return 0;
}

