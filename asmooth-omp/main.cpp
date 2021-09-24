#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "reference.cpp"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
    exit(1);
  }

  // only a square image is supported
  const int Lx = atoi(argv[1]);
  const int Ly = Lx;
  const int size = Lx * Ly;

  const int Threshold = atoi(argv[2]);
  const int MaxRad = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  // input image
  float *img = (float*) malloc (sizeof(float) * size);

  // host and device results
  float *norm = (float*) malloc (sizeof(float) * size);
  float *h_norm = (float*) malloc (sizeof(float) * size);

  int *box = (int*) malloc (sizeof(int) * size);
  int *h_box = (int*) malloc (sizeof(int) * size);

  float *out = (float*) malloc (sizeof(float) * size);
  float *h_out = (float*) malloc (sizeof(float) * size);

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = rand() % 256;
    norm[i] = box[i] = out[i] = 0;
  }

  #pragma omp target data map(alloc: img[0:size], norm[0:size], box[0:size]) \
                          map(to: out[0:size]) 
  {
    for (int i = 0; i < repeat; i++) {
      // restore input image
      #pragma omp target update to(img[0:size])
      // reset norm
      #pragma omp target update to(norm[0:size])

      // launch three kernels
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
      for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
          float sum = 0.f;
          int s = 1;
          int q = 1;
          int ksum = 0;

          while (sum < Threshold && q < MaxRad) {
            s = q;
            sum = 0.f;
            ksum = 0;

            for (int i = -s; i < s+1; i++)
              for (int j = -s; j < s+1; j++)
                if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
                  sum += img[(x+i)*Ly+y+j];
                  ksum++;
                }
            q++;
          }

          box[x*Ly+y] = s;  // save the box size

          for (int i = -s; i < s+1; i++)
            for (int j = -s; j < s+1; j++)
              if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly)
                if (ksum != 0) {
                  #pragma omp atomic update
                  norm[(x+i)*Ly+y+j] += 1.f / (float)ksum;
                }
        }
      }

      // normalize the image
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
      for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++) 
          if (norm[x*Ly+y] != 0) img[x*Ly+y] /= norm[x*Ly+y];

      // output file
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
      for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
          int s = box[x*Ly+y];
          float sum = 0.f;
          int ksum = 0;

          // resmooth with normalized image
          for (int i = -s; i < s+1; i++)
            for (int j = -s; j < s+1; j++) {
              if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
                sum += img[(x+i)*Ly+y+j];
                ksum++;
              }
            }
          if (ksum != 0) out[x*Ly+y] = sum / (float)ksum;
        }
      }
    }

    #pragma omp target update from(out[0:size])
    #pragma omp target update from(box[0:size])
    #pragma omp target update from(norm[0:size])
  }

  // verify
  reference (Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  free(img);
  free(norm);
  free(h_norm);
  free(box);
  free(h_box);
  free(out);
  free(h_out);
  return 0;
}
