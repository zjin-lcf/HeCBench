#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "reference.h"

void entropy(
  nd_item<2> &item,
      float *__restrict d_entropy,
  const char*__restrict d_val, 
  int height, int width)
{
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
        count[d_val[yy * width + xx]]++;
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

  if(y < height && x < width) d_entropy[y * width + x] = entropy;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <width> <height>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]); 
  const int height = atoi(argv[2]); 

  const int input_bytes = width * height * sizeof(char);
  const int output_bytes = width * height * sizeof(float);
  char* input = (char*) malloc (input_bytes);
  float* output = (float*) malloc (output_bytes);
  float* output_ref = (float*) malloc (output_bytes);

  srand(123);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      input[i * width + j] = rand() % 16;

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<char, 1> d_input (input, width * height);
  buffer<float, 1> d_output (output, width * height);

  range<2> gws ((height+15)/16*16, (width+15)/16*16);
  range<2> lws (16, 16);

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto in = d_input.get_access<sycl_read>(cgh);
      auto out = d_output.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class base>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        entropy (item, out.get_pointer(), in.get_pointer(), height, width);
      });
    });

  float logTable[26];
  for (int i = 0; i <= 25; i++) logTable[i] = i <= 1 ? 0 : i*log2f(i);
  buffer<float, 1> d_logTable (logTable, 26);

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto input = d_input.get_access<sycl_read>(cgh);
      auto logTable = d_logTable.get_access<sycl_read>(cgh);
      auto output = d_output.get_access<sycl_discard_write>(cgh);
      accessor<char, 2, sycl_read_write, access::target::local> sd_count ({16, 256}, cgh);
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
              sd_count[input[yy*width+xx]][idx]++;
              total++;
            }
          }
        }

        float entropy = 0;
        for(int k = 0; k < 16; k++)
          entropy -= logTable[sd_count[k][idx]];
        
        entropy = entropy / total + sycl::log2((float)total);
        if(y < height && x < width) output[y*width+x] = entropy;
      });
    });
  }

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
