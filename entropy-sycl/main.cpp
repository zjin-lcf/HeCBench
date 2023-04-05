#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"
#include "reference.h"

void entropy(
  nd_item<2> &item,
      float *__restrict d_entropy,
  const char*__restrict d_val, 
  int height, int width)
{
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]); 
  const int height = atoi(argv[2]); 
  const int repeat = atoi(argv[3]); 

  const int input_bytes = width * height * sizeof(char);
  const int output_bytes = width * height * sizeof(float);
  char* input = (char*) malloc (input_bytes);
  float* output = (float*) malloc (output_bytes);
  float* output_ref = (float*) malloc (output_bytes);

  srand(123);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      input[i * width + j] = rand() % 16;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  char* d_input = malloc_device<char>(width * height, q);
  q.memcpy(d_input, input, input_bytes);

  float* d_output = malloc_device<float>(width * height, q);

  range<2> gws ((height+15)/16*16, (width+15)/16*16);
  range<2> lws (16, 16);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class base>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        const int x = item.get_global_id(1);
        const int y = item.get_global_id(0);

        // value of matrix element ranges from 0 inclusive to 16 exclusive
        char count[16];
        for (int i = 0; i < 16; i++) count[i] = 0;

        // total number of valid elements
        char total = 0;

        // 5x5 window
        for(int dy = -2; dy <= 2; dy++) {
          for(int dx = -2; dx <= 2; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if(xx >= 0 && yy >= 0 && yy < height && xx < width) {
              count[d_input[yy * width + xx]]++;
              total++;
            }
          }
        }

        float entropy = 0;
        if (total < 1) {
          total = 1;
        } else {
          for(int k = 0; k < 16; k++) {
            float p = sycl::native::divide((float)count[k], (float)total);
            entropy -= p * sycl::log2(p);
          }
        }

        if(y < height && x < width) d_output[y * width + x] = entropy;
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (baseline) execution time %f (s)\n", (time * 1e-9f) / repeat);

  float logTable[26];
  for (int i = 0; i <= 25; i++) logTable[i] = i <= 1 ? 0 : i*log2f(i);

  float *d_logTable = malloc_device<float>(26, q);
  q.memcpy(d_logTable, logTable, 26 * sizeof(float));

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      accessor<int, 2, sycl_read_write, access::target::local> sd_count ({16, 256}, cgh);
      cgh.parallel_for<class opt>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        const int x = item.get_global_id(1);
        const int y = item.get_global_id(0);
        const int idx = item.get_local_id(0)*16 + item.get_local_id(1);

        for(int i = 0; i < 16;i++) sd_count[i][idx] = 0;

        char total = 0;
        for(int dy = -2; dy <= 2; dy++) {
          for(int dx = -2; dx <= 2; dx++) {
            int xx = x + dx,
                yy = y + dy;

            if(xx >= 0 && yy >= 0 && yy < height && xx < width) {
              sd_count[d_input[yy*width+xx]][idx]++;
              total++;
            }
          }
        }

        float entropy = 0;
        for(int k = 0; k < 16; k++)
          entropy -= d_logTable[sd_count[k][idx]];
        
        entropy = entropy / total + sycl::log2((float)total);
        if(y < height && x < width) d_output[y*width+x] = entropy;
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (optimized) execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(output, d_output, output_bytes).wait();

  free(d_input, q);
  free(d_output, q);
  free(d_logTable, q);

  // verify
  reference(output_ref, input, height, width);

  bool ok = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (fabsf(output[i * width + j] - output_ref[i * width + j]) > 1e-3f) {
        ok = false; 
        break;
      }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
 
  free(input);
  free(output);
  free(output_ref);
  return 0;
}
