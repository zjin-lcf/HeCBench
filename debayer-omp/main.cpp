#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "util.h"
#include "image.h"
#include "kernel.h"

int main(int argc, char* argv[]) 
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_image_pitch = width;
  const int output_image_pitch = width * 4;
  const int numPix = width * height;
  const int input_image_size = numPix * sizeof(uchar);
  const int output_image_size = numPix * 4 * sizeof(uchar);

  uchar *input = (uchar*) malloc (input_image_size);
  uchar *output = (uchar*) malloc (output_image_size);

  // RGGB, GRBG, GBRG, BGGR
  const int bayer_pattern = RGGB;

  // input image with random values
  srand(123);
  for (int i = 0; i < numPix; i++) {
    input[i] = rand() % 256;
  }

  const uint teamX = (width + tile_cols - 1) / tile_cols;
  const uint teamY = (height + tile_rows - 1) / tile_rows;

  #pragma omp target data map(to: input[0:numPix]) map(from: output[0:numPix])
  {

    //this version takes a tile (z=1) and each tile job does 4 line median sorts
    for (int i = 0; i < repeat; i++) 
      malvar_he_cutler_demosaic (
        teamX, teamY,
        height, width, 
        input, input_image_pitch,
        output, output_image_pitch,
        bayer_pattern );
  }

  long sum = 0;
  for (int i = 0; i < numPix; i++) sum += output[i];
  printf("Checksum: %ld\n", sum);

  free(input);
  free(output);
  return 0;
}
