#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "reference.h"

void entropy(
        float *__restrict d_entropy,
    const char*__restrict d_val, 
    int height, int width)
{
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) 
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
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
          float p = (float)count[k] / (float)total;
          entropy -= p * log2f(p);
        }
      }

      d_entropy[y * width + x] = entropy;
    }
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

  #pragma omp target data map(to: input[0:width*height]) \
                          map(from: output[0:width*height])
  {
    for (int i = 0; i < 100; i++)
      entropy(output, input, height, width);
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
