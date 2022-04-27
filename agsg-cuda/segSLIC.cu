#include <stdio.h>
#include <string.h>
#include "segSLIC.h"
#include "util.h"

extern int SPSV;

__global__ void kUpdateClusterCentersImproved(
   const float4* __restrict__ floatBuffer,
   const int* __restrict__ maskBuffer,
   int nWidth, int nHeight, int clusterSize,
   SLICClusterCenter* vSLICCenterList)
{
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  __shared__ float3 avLab[BLOCK_SIZE1];
  __shared__ int2 avXY[BLOCK_SIZE1];
  __shared__ unsigned int npoints[BLOCK_SIZE1];

  int yBegin = vSLICCenterList[bid].xy.y - clusterSize;
  int yEnd = vSLICCenterList[bid].xy.y + clusterSize;
  int xBegin = vSLICCenterList[bid].xy.x - clusterSize;
  int xEnd = vSLICCenterList[bid].xy.x + clusterSize;

  yBegin = yBegin < 0 ? 0 : yBegin;
  xBegin = xBegin < 0 ? 0 : xBegin;
  yEnd = yEnd < nHeight ? yEnd : nHeight; 
  xEnd = xEnd < nWidth ? xEnd : nWidth;

  int areaSize = xEnd - xBegin;
  if(areaSize == 0)
  {
    return;
  }

  int yStep = tid / areaSize;
  int xStep = tid - yStep*areaSize; //int xStep = tid % areaSize;
  int xOffset = xBegin + xStep;
  int yOffset = yBegin + yStep;

  float3 lab = make_float3(0.f, 0.f, 0.f);
  int2 xy = make_int2(0, 0);
  int p = 0;
  while (yOffset < yEnd) 
  {
    int offset = yOffset*nWidth+xOffset;
    if(maskBuffer[offset] == blockIdx.x) 
    {
      lab.x += floatBuffer[offset].x;
      lab.y += floatBuffer[offset].y;
      lab.z += floatBuffer[offset].z;
      xy.x += xOffset;
      xy.y += yOffset;
      p += 1;
    }
    yStep = (xStep + blockDim.x) / areaSize;
    xStep = (xStep + blockDim.x) - areaSize*yStep; // xStep = (xStep + blockDim.x) % areaSize;
    xOffset = xBegin + xStep;
    yOffset += yStep;
  }

  avLab[tid].x = lab.x;
  avLab[tid].y = lab.y;
  avLab[tid].z = lab.z;
  avXY[tid].x = xy.x;
  avXY[tid].y = xy.y;
  npoints[tid] = p;
  __syncthreads();

  // BLOCK_SIZE1 have to be power of two  
#pragma unroll
  for (unsigned int s=BLOCK_SIZE1>>1; s>0; s>>=1) 
  {
    if (tid < s)
    {
      avLab[tid].x += avLab[tid + s].x;
      avLab[tid].y += avLab[tid + s].y;
      avLab[tid].z += avLab[tid + s].z;
      avXY[tid].x += avXY[tid + s].x;
      avXY[tid].y += avXY[tid + s].y;
      npoints[tid] += npoints[tid + s];
    }
    __syncthreads();
  }

  if(tid == 0)
  {
    vSLICCenterList[blockIdx.x].lab.x = avLab[0].x / (float)npoints[0];
    vSLICCenterList[blockIdx.x].lab.y = avLab[0].y / (float)npoints[0];
    vSLICCenterList[blockIdx.x].lab.z = avLab[0].z / (float)npoints[0];
    vSLICCenterList[blockIdx.x].xy.x = avXY[0].x / (float)npoints[0];
    vSLICCenterList[blockIdx.x].xy.y = avXY[0].y / (float)npoints[0];
    vSLICCenterList[blockIdx.x].nPoints = npoints[0];
  }

  __syncthreads();


}

__global__ void kInitClusterCentersNative(
  const float4* __restrict__ floatBuffer,
  int nWidth, int nHeight, int nSegs,
  SLICClusterCenter* __restrict__ vSLICCenterList )
{

  int blockWidth=nWidth/blockDim.x;
  int blockHeight=nHeight/gridDim.x;

  int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
  int offsetBlock = blockIdx.x * blockHeight * nWidth + threadIdx.x * blockWidth;

  float2 avXY;

  avXY.x=threadIdx.x*blockWidth + (float)blockWidth/2.f;
  avXY.y=blockIdx.x*blockHeight + (float)blockHeight/2.f;

  //use a single point to init center
  int offset=offsetBlock + blockHeight/2 * nWidth+ blockWidth/2 ;

  float4 fPixel=floatBuffer[offset];

  vSLICCenterList[clusterIdx].lab=fPixel;
  vSLICCenterList[clusterIdx].xy=avXY;
  vSLICCenterList[clusterIdx].nPoints=0;

}

__global__ void kInitClusterCentersImproved(
  const float4* __restrict__ floatBuffer,
  int nWidth, int nHeight, int nSegs,
  SLICClusterCenter* __restrict__ vSLICCenterList )
{

  int blockWidth=nWidth/blockDim.x;
  int blockHeight=nHeight/gridDim.x;

  int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
  int offsetBlock = blockIdx.x * blockHeight * nWidth + threadIdx.x * blockWidth;

  float2 avXY;

  avXY.x=threadIdx.x*blockWidth + blockWidth*0.5f;
  avXY.y=blockIdx.x*blockHeight + blockHeight*0.5f;

  //use a single point to init center
  int offset=offsetBlock + blockHeight/2 * nWidth+ blockWidth/2 ;

  float4 fPixel=floatBuffer[offset];

  vSLICCenterList[clusterIdx].lab=fPixel;
  vSLICCenterList[clusterIdx].xy=avXY;
  vSLICCenterList[clusterIdx].nPoints=0;

}

__global__ void kIterateKmeansNative(
  int* __restrict__ maskBuffer,
  const float4* __restrict__ floatBuffer, 
  int nWidth, int nHeight, int nSegs, int nClusterIdxStride, 
  SLICClusterCenter* __restrict__ vSLICCenterList, 
  bool bLabelImg, float weight)
{

  //for reading cluster centers
  __shared__ float4 fShareLab[3][3];
  __shared__ float2 fShareXY[3][3];

  //pixel index
  __shared__ SLICClusterCenter pixelUpdateList[BLOCK_SIZE1];
  __shared__ float2 pixelUpdateIdx[BLOCK_SIZE1];


  int clusterIdx=blockIdx.y;
  int blockCol=clusterIdx%nClusterIdxStride;
  int blockRow=clusterIdx/nClusterIdxStride;
  //int upperBlockHeight=blockDim.y*gridDim.x;

  int lowerBlockHeight=blockDim.y;
  int blockWidth=blockDim.x;
  int upperBlockHeight=blockWidth;

  int innerBlockHeightIdx=lowerBlockHeight*blockIdx.x+threadIdx.y;

  float M=weight;
  float invWeight=1/((blockWidth/M)*(blockWidth/M));

  int offsetBlock = (blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight)*nWidth+blockCol*blockWidth;
  int offset=offsetBlock+threadIdx.x+threadIdx.y*nWidth;

  int rBegin=(blockRow>0)?0:1;
  int rEnd=(blockRow+1>(gridDim.y/nClusterIdxStride-1))?1:2;
  int cBegin=(blockCol>0)?0:1;
  int cEnd=(blockCol+1>(nClusterIdxStride-1))?1:2;

  if (threadIdx.x<3 && threadIdx.y<3)
  {
    if (threadIdx.x>=cBegin && threadIdx.x<=cEnd && threadIdx.y>=rBegin && threadIdx.y<=rEnd)
    {
      int cmprIdx=(blockRow+threadIdx.y-1)*nClusterIdxStride+(blockCol+threadIdx.x-1);
      fShareLab[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].lab;
      fShareXY[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].xy;
    }
  }
  __syncthreads();

  if (innerBlockHeightIdx>=blockWidth)
  {
    return;
  }

  if (offset>=nWidth*nHeight)
  {
    return;
  }

  // finding the nearest center for current pixel
  float fY=blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight+threadIdx.y;
  float fX=blockCol*blockWidth+threadIdx.x;

  if (fY<nHeight && fX<nWidth)
  {
    float4 fPoint=floatBuffer[offset];
    float minDis=9999;
    int nearestCenter=-1;
    int nearestR, nearestC;

    for (int r=rBegin;r<=rEnd;r++)
    {
      for (int c=cBegin;c<=cEnd;c++)
      {
        int cmprIdx=(blockRow+r-1)*nClusterIdxStride+(blockCol+c-1);

        //compute SLIC distance
        float fDab=(fPoint.x-fShareLab[r][c].x)*(fPoint.x-fShareLab[r][c].x)
          +(fPoint.y-fShareLab[r][c].y)*(fPoint.y-fShareLab[r][c].y)
          +(fPoint.z-fShareLab[r][c].z)*(fPoint.z-fShareLab[r][c].z);

        float fDxy=(fX-fShareXY[r][c].x)*(fX-fShareXY[r][c].x)
          +(fY-fShareXY[r][c].y)*(fY-fShareXY[r][c].y);

        float fDis=fDab+invWeight*fDxy;

        if (fDis<minDis)
        {
          minDis=fDis;
          nearestCenter=cmprIdx;
          nearestR=r;
          nearestC=c;
        }

      }
    }

    if (nearestCenter>-1)
    {
      int pixelIdx=threadIdx.y*blockWidth+threadIdx.x;

      pixelUpdateList[pixelIdx].lab=fPoint;
      pixelUpdateList[pixelIdx].xy.x=fX;
      pixelUpdateList[pixelIdx].xy.y=fY;

      pixelUpdateIdx[pixelIdx].x=nearestC;
      pixelUpdateIdx[pixelIdx].y=nearestR;

      if (bLabelImg)
      {
        maskBuffer[offset]=nearestCenter;
      }
    }
  }
  else
  {
    int pixelIdx=threadIdx.y*blockWidth+threadIdx.x;

    pixelUpdateIdx[pixelIdx].x=-1;
    pixelUpdateIdx[pixelIdx].y=-1;

  }
  __syncthreads();
}

__global__ void kIterateKmeansImproved(
  int* __restrict__ maskBuffer,
  const float4* __restrict__ floatBuffer, 
  int nWidth, int nHeight, int nSegs,  int nClusterIdxStride, 
  const SLICClusterCenter* __restrict__ vSLICCenterList, 
  bool bLabelImg, float weight)
{
  //for reading cluster centers
  __shared__ float4 fShareLab[3][3];
  __shared__ float2 fShareXY[3][3];

  int clusterIdx=blockIdx.y;
  //int blockCol=clusterIdx%nClusterIdxStride;
  int blockRow=clusterIdx/nClusterIdxStride;
  int blockCol=blockIdx.y - blockRow*nClusterIdxStride;

  int lowerBlockHeight=blockDim.y;
  int blockWidth=blockDim.x;
  int upperBlockHeight=blockWidth;

  int innerBlockHeightIdx=lowerBlockHeight*blockIdx.x+threadIdx.y;

  float M=weight;
  float BM = blockWidth/M;
  float invWeight=1/(BM*BM);

  int offsetBlock = (blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight)*nWidth+blockCol*blockWidth;
  int offset=offsetBlock+threadIdx.x+threadIdx.y*nWidth;

  int rBegin=(blockRow>0)?0:1;
  int rEnd=(blockRow+1>(gridDim.y/nClusterIdxStride-1))?1:2;
  int cBegin=(blockCol>0)?0:1;
  int cEnd=(blockCol+1>(nClusterIdxStride-1))?1:2;

  if (threadIdx.x<3 && threadIdx.y<3)
  {
    if (threadIdx.x>=cBegin && threadIdx.x<=cEnd && threadIdx.y>=rBegin && threadIdx.y<=rEnd)
    {
      int cmprIdx=(blockRow+threadIdx.y-1)*nClusterIdxStride+(blockCol+threadIdx.x-1);
      fShareLab[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].lab;
      fShareXY[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].xy;
    }
  }
  __syncthreads();

  if (innerBlockHeightIdx>=blockWidth)
  {
    return;
  }

  if (offset>=nWidth*nHeight)
  {
    return;
  }

  // finding the nearest center for current pixel
  float fY=blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight+threadIdx.y;
  float fX=blockCol*blockWidth+threadIdx.x;

  if (fY<nHeight && fX<nWidth)
  {
    float4 fPoint=floatBuffer[offset];
    float minDis=9999;
    int nearestCenter=-1;

    for (int r=rBegin;r<=rEnd;r++)
    {
      for (int c=cBegin;c<=cEnd;c++)
      {
        int cmprIdx=(blockRow+r-1)*nClusterIdxStride+(blockCol+c-1);

        //compute SLIC distance
        float3 labDif = {fPoint.x-fShareLab[r][c].x,fPoint.y-fShareLab[r][c].y,fPoint.z-fShareLab[r][c].z};
        float fDab= labDif.x*labDif.x + labDif.y*labDif.y + labDif.z*labDif.z;
        +(fPoint.y-fShareLab[r][c].y)*(fPoint.y-fShareLab[r][c].y)
          +(fPoint.z-fShareLab[r][c].z)*(fPoint.z-fShareLab[r][c].z);

        float2 xyDif = {fX-fShareXY[r][c].x,fY-fShareXY[r][c].y};
        float fDxy = (xyDif.x*xyDif.x+xyDif.y*xyDif.y);

        float fDis=fDab+invWeight*fDxy;

        if (fDis<minDis)
        {
          minDis=fDis;
          nearestCenter=cmprIdx;
        }

      }
    }

    if (nearestCenter>-1)
    {
      if (bLabelImg)
      {
        maskBuffer[offset]=nearestCenter;
      }
    }
  }
  __syncthreads();
}

__global__ void kUpdateClusterCentersNative(
  const float4* __restrict__ floatBuffer,
  int* __restrict__ maskBuffer,
  int nWidth, int nHeight, int nSegs,
  SLICClusterCenter* __restrict__ vSLICCenterList )
{

  int blockWidth=nWidth/blockDim.x;
  int blockHeight=nHeight/gridDim.x;

  int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
  int offsetBlock = threadIdx.x * blockWidth+ blockIdx.x * blockHeight * nWidth;

  float2 crntXY=vSLICCenterList[clusterIdx].xy;
  float4 avLab;
  float2 avXY;
  int nPoints=0;

  avLab.x=0;
  avLab.y=0;
  avLab.z=0;

  avXY.x=0;
  avXY.y=0;

  int yBegin=0 < (crntXY.y - blockHeight) ? (crntXY.y - blockHeight) : 0;
  int yEnd= nHeight > (crntXY.y + blockHeight) ? (crntXY.y + blockHeight) : (nHeight - 1);  
  int xBegin=0 < (crntXY.x - blockWidth) ? (crntXY.x - blockWidth) : 0;
  int xEnd= nWidth > (crntXY.x + blockWidth) ? (crntXY.x + blockWidth) : (nWidth - 1);

  //update to cluster centers
  for (int i = yBegin; i < yEnd ; i++)
  {
    for (int j = xBegin; j < xEnd; j++)
    {
      int offset=j + i * nWidth;      
      float4 fPixel=floatBuffer[offset];
      int pIdx=maskBuffer[offset];

      if (pIdx==clusterIdx)
      {
        avLab.x+=fPixel.x;
        avLab.y+=fPixel.y;
        avLab.z+=fPixel.z;

        avXY.x+=j;
        avXY.y+=i;

        nPoints++;
      }
    }
  }

  avLab.x/=nPoints;
  avLab.y/=nPoints;
  avLab.z/=nPoints;

  avXY.x/=nPoints;
  avXY.y/=nPoints;

  vSLICCenterList[clusterIdx].lab=avLab;
  vSLICCenterList[clusterIdx].xy=avXY;
  vSLICCenterList[clusterIdx].nPoints=nPoints;
}

void SLICImgSeg(
  int* maskBuffer, float4* floatBuffer, 
  int nWidth, int nHeight, int nSegs, int nIt, 
  SLICClusterCenter* vSLICCenterList, float weight)
{
  int nClusterSize=(int)sqrt((float)iDivUp(nWidth*nHeight,nSegs));
  int nClustersPerCol=iDivUp(nHeight,nClusterSize);
  int nClustersPerRow=iDivUp(nWidth,nClusterSize);
  int nBlocksPerCluster=iDivUp(nClusterSize*nClusterSize,BLOCK_SIZE1);

  int nSeg=nClustersPerCol*nClustersPerRow;

  int nBlockWidth=nClusterSize;
  int nBlockHeight=iDivUp(nClusterSize,nBlocksPerCluster);

  int areaSize = nClusterSize*nClusterSize*4;

  dim3 ThreadPerBlock_init(nClustersPerRow); //x
  dim3 BlockPerGrid_init(nClustersPerCol); //y

  dim3 ThreadPerBlock(nBlockWidth,nBlockHeight);
  dim3 BlockPerGrid(nBlocksPerCluster,nSeg);

  dim3 ThreadPerBlockUpdate(BLOCK_SIZE1);
  dim3 BlockPerGridUpdate(nSeg);

  switch (SPSV)
  {
    // orginal gSlic
    case 1:
      kInitClusterCentersNative<<<BlockPerGrid_init,ThreadPerBlock_init>>>(
        floatBuffer,nWidth,nHeight,nSegs,vSLICCenterList);

      //5 iterations have already given good result
      for (int i=0;i<nIt;i++)
      {
        kIterateKmeansNative<<<BlockPerGrid,ThreadPerBlock>>>(
          maskBuffer,floatBuffer,nWidth,nHeight,nSeg,nClustersPerRow,vSLICCenterList,true, weight);

        kUpdateClusterCentersNative<<<BlockPerGrid_init,ThreadPerBlock_init>>>(
          floatBuffer,maskBuffer,nWidth,nHeight,nSeg,vSLICCenterList);
      }
      kIterateKmeansNative<<<BlockPerGrid,ThreadPerBlock>>>(
        maskBuffer,floatBuffer,nWidth,nHeight,nSeg,nClustersPerRow,vSLICCenterList,true, weight);
      break;

      // improved gSLIC
    case 2:
      kInitClusterCentersImproved<<<BlockPerGrid_init,ThreadPerBlock_init>>>(
        floatBuffer,nWidth,nHeight,nSegs,vSLICCenterList);

      //5 iterations have already given good result
      cudaFuncSetCacheConfig(kIterateKmeansImproved,cudaFuncCachePreferL1);
      for (int i=0;i<nIt;i++)
      {
        kIterateKmeansImproved<<<BlockPerGrid,ThreadPerBlock>>>(
          maskBuffer,floatBuffer,nWidth,nHeight,nSeg,nClustersPerRow,vSLICCenterList,true, weight);

        kUpdateClusterCentersImproved<<<BlockPerGridUpdate,ThreadPerBlockUpdate>>>(
          floatBuffer, maskBuffer, nWidth, nHeight, nClusterSize, vSLICCenterList);
      }
      kIterateKmeansImproved<<<BlockPerGrid,ThreadPerBlock>>>(
        maskBuffer,floatBuffer,nWidth,nHeight,nSeg,nClustersPerRow,vSLICCenterList,true, weight);
      break;
    default:
      break;
  }  
}

