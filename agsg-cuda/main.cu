#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include "util.h"
#include "imgTrans.h"
#include "segSLIC.h"
#include "imgTrans.h"

int SPSV;  // SuperpixelSegmentation: 1 - gSLIC-native, 2 - gSLIC-improved

int main() {

  const int width = 3840;
  const int height = 2160;
  const float weight = 10.f;
  const int nIt = 100;

  // actual number of segments will be adjusted with the number of clusters
  int nSeg = 1200;  

  const int imgSize = height*width;
  uchar4 *imgPixels = (uchar4*) malloc (imgSize*sizeof(uchar4));

  std::mt19937 gen(19937);
  std::uniform_int_distribution<unsigned char> dis(0, 255);
  for (int i = 0; i < imgSize; i++) {
    imgPixels[i] = make_uchar4(dis(gen), dis(gen), dis(gen), dis(gen)); 
  }

  int *maskPixels = (int*) malloc (imgSize*sizeof(int));

  uchar4* rgbBuffer;
  float4* floatBuffer;
  int* maskBuffer;
  cudaMalloc((void**) &rgbBuffer, imgSize*sizeof(uchar4));
  cudaMalloc((void**) &floatBuffer, imgSize*sizeof(float4));
  cudaMalloc((void**) &maskBuffer, imgSize*sizeof(int));

  // for SLICC 
  int nClusterSize=(int)sqrt((float)iDivUp(imgSize,nSeg));
  int nClustersPerCol=iDivUp(height,nClusterSize);
  int nClustersPerRow=iDivUp(width,nClusterSize);
  int nBlocksPerCluster=iDivUp(nClusterSize*nClusterSize,BLOCK_SIZE1);

  nSeg=nClustersPerCol*nClustersPerRow;

  SLICClusterCenter* sliccCenterList;

  int nMaxSegs=iDivUp(width,BLOCK_SIZE2)*iDivUp(height,BLOCK_SIZE2);
  cudaMalloc((void**) &sliccCenterList,nMaxSegs*sizeof(SLICClusterCenter));

  for (int method = 1; method <= 3; method++) {

    for (SPSV = 1; SPSV <= 2; SPSV++) {  // cudaSegSLIC.cu

      cudaMemset(floatBuffer,0,imgSize*sizeof(float4));
      cudaMemset(maskBuffer,0,imgSize*sizeof(int));
      cudaMemset(sliccCenterList,0,nMaxSegs*sizeof(SLICClusterCenter));

      cudaDeviceSynchronize();
      auto start = std::chrono::steady_clock::now();

      cudaMemcpy(rgbBuffer,imgPixels,imgSize*sizeof(uchar4),cudaMemcpyHostToDevice);

      switch (method)
      {
        case 1:
          Rgb2CIELab(rgbBuffer,floatBuffer,width,height); break;

        case 2:
          Rgb2XYZ(rgbBuffer,floatBuffer,width,height); break;

        case 3:
          Uchar4ToFloat4(rgbBuffer,floatBuffer,width,height); break;
      }

      SLICImgSeg(maskBuffer,floatBuffer,width,height,nSeg,nIt,sliccCenterList,weight);

      cudaMemcpy(maskPixels,maskBuffer,imgSize*sizeof(int),cudaMemcpyDeviceToHost);

      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<float> time = end - start;

      printf("%s: ", SPSV == 1 ? "gSLIC-base" : "gSLIC-improved");   
      printf("device offload time is %f (s)\n", time.count());

      long checkSum = 0;
      for (int i = 0; i < imgSize; i++) {
         checkSum += maskPixels[i];
      }
      printf("mask checksum (method %d) = %ld\n\n", method, checkSum);
    }
  }

  cudaFree(rgbBuffer);
  cudaFree(floatBuffer);
  cudaFree(maskBuffer);
  cudaFree(sliccCenterList);
  free(maskPixels);
  free(imgPixels);
  return 0;
}

