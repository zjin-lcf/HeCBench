#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

#define BLOCK_SIZE 256

// begin of findMovingPixels
void findMovingPixels(
  const int numTeams,
  const int numThreads,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Img1,
  const unsigned char *__restrict Img2,
  const unsigned char *__restrict Tn,
        unsigned char *__restrict Mp) // moving pixel map
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (size_t i = 0; i < imgSize; i++) {
    if ( abs(Img[i] - Img1[i]) > Tn[i] || abs(Img[i] - Img2[i]) > Tn[i] )
      Mp[i] = 255;
    else
      Mp[i] = 0;
  }
}
// end of findMovingPixels

// alpha = 0.92
// begin of updateBackground
void updateBackground(
  const int numTeams,
  const int numThreads,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Mp,
        unsigned char *__restrict Bn)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (size_t i = 0; i < imgSize; i++) {
    if ( Mp[i] == 0 ) Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];
  }
}
// end of updateBackground

// alpha = 0.92, c = 3
// begin of updateThreshold
void updateThreshold(
  const int numTeams,
  const int numThreads,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Mp,
  const unsigned char *__restrict Bn,
        unsigned char *__restrict Tn)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (size_t i = 0; i < imgSize; i++) {
    if (Mp[i] == 0) {
      float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
      Tn[i] = fmaxf(th, 20.f);
    }
  }
}
// end of updateThreshold

//
// merge three kernels into a single kernel
//
// begin of merge
void merge(
  const int numTeams,
  const int numThreads,
  const size_t imgSize,
  const unsigned char *__restrict Img,
  const unsigned char *__restrict Img1,
  const unsigned char *__restrict Img2,
        unsigned char *__restrict Tn,
        unsigned char *__restrict Bn)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (size_t i = 0; i < imgSize; i++) {
    if ( abs(Img[i] - Img1[i]) <= Tn[i] && abs(Img[i] - Img2[i]) <= Tn[i] ) {
      // update background
      Bn[i] = 0.92 * Bn[i] + 0.08 * Img[i];

      // update threshold
      float th = 0.92 * Tn[i] + 0.24 * (Img[i] - Bn[i]);
      Tn[i] = fmaxf(th, 20.f);
    }
  }
}
// end of merge

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <merge> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int merged = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int imgSize = width * height;
  const size_t imgSize_bytes = imgSize * sizeof(unsigned char);
  unsigned char *Img = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Img1 = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Img2 = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Bn = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Bn_ref = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Mp = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Tn = (unsigned char*) malloc (imgSize_bytes);
  unsigned char *Tn_ref = (unsigned char*) malloc (imgSize_bytes);

  std::mt19937 generator( 123 );
  std::uniform_int_distribution<int> distribute( 0, 255 );

  for (int j = 0; j < imgSize; j++) {
    Bn_ref[j] = Bn[j] = distribute(generator);
    Tn_ref[j] = Tn[j] = 128;
  }

  const int numTeams = (imgSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int numThreads = BLOCK_SIZE;

  long time = 0;

  #pragma omp target data map (tofrom: Bn[0:imgSize]) \
                          map (tofrom: Tn[0:imgSize]) \
                          map (alloc: Mp[0:imgSize], \
                                       Img[0:imgSize], \
                                      Img1[0:imgSize], \
                                      Img2[0:imgSize])
  {
    for (int i = 0; i < repeat; i++) {

      for (int j = 0; j < imgSize; j++) {
        Img[j] = distribute(generator);
      }
   
      #pragma omp target update to (Img[0:imgSize])

    // Time t   : Image   | Image1   | Image2
    // Time t+1 : Image2  | Image    | Image1
    // Time t+2 : Image1  | Image2   | Image
      unsigned char *t = Img2;
      Img2 = Img1;
      Img1 = Img;
      Img = t;

      if (i >= 2) {
        if (merged) {
          auto start = std::chrono::steady_clock::now();
          merge ( numTeams, numThreads, imgSize, Img, Img1, Img2, Tn, Bn );
          auto end = std::chrono::steady_clock::now();
          time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        else {
          auto start = std::chrono::steady_clock::now();
          findMovingPixels ( numTeams, numThreads, imgSize, Img, Img1, Img2, Tn, Mp );
          updateBackground ( numTeams, numThreads, imgSize, Img, Mp, Bn );
          updateThreshold ( numTeams, numThreads, imgSize, Img, Mp, Bn, Tn );
          auto end = std::chrono::steady_clock::now();
          time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        merge_ref ( imgSize, Img, Img1, Img2, Tn_ref, Bn_ref );
      }
    }

    float kernel_time = (repeat <= 2) ? 0 : (time * 1e-3f) / (repeat - 2);
    printf("Average kernel execution time: %f (us)\n", kernel_time);
  }

  // verification
  int max_error = 0;
  for (int i = 0; i < imgSize; i++) {
    if (abs(Tn[i] - Tn_ref[i]) > max_error)
      max_error = abs(Tn[i] - Tn_ref[i]);
  }
  for (int i = 0; i < imgSize; i++) {
    if (abs(Bn[i] - Bn_ref[i]) > max_error)
      max_error = abs(Bn[i] - Bn_ref[i]);
  }
  printf("Max error is %d\n", max_error);

  printf("%s\n", max_error ? "FAIL" : "PASS");

  free(Img);
  free(Img1);
  free(Img2);
  free(Tn);
  free(Bn);
  free(Tn_ref);
  free(Bn_ref);
  free(Mp);

  return 0;
}
