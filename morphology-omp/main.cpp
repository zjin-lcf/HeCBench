#include <cstdio>
#include <cstdlib>
#include "morphology.h"

void display(unsigned char *img, const int height, const int width)
{
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++)
      printf("%d ", img[i*width+j]);
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
  int hsize = atoi(argv[1]);  // kernel width (32 is maximum without adjusting the share memory size)
  int vsize = atoi(argv[2]);  // kernel height (32 is maximum without adjusting the share memory size)
  int width = atoi(argv[3]);  // image width
  int height = atoi(argv[4]); // image height

  unsigned int memSize = width * height * sizeof(unsigned char);

  unsigned char* srcImg = (unsigned char*) malloc (memSize);
  unsigned char* tmpImg = (unsigned char*) malloc (memSize);

  for (int i = 0; i < height; i++) 
    for (int j = 0; j < width; j++)
      srcImg[i*width+j] = (i == (height/2 - 1) && 
                           j == (width/2 - 1)) ? WHITE : BLACK;

#pragma omp target data map(tofrom: srcImg[0:memSize]) \
                        map(alloc: tmpImg[0:memSize])
{
  for (int n = 0; n < 1; n++) {
    dilate(srcImg, tmpImg, width, height, hsize, vsize);
    erode(srcImg, tmpImg, width, height, hsize, vsize);
  }
}

  int s = 0;
  for (unsigned int i = 0; i < memSize; i++) s += srcImg[i];
  printf("%s\n", s == WHITE ? "PASS" : "FAIL");

  free(srcImg);
  free(tmpImg);
  return 0;
}
