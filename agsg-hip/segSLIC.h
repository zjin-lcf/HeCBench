#ifndef __SEG_SLIC__
#define __SEG_SLIC__

typedef struct
{
  float4 lab;
  float2 xy;
  int nPoints;
} SLICClusterCenter;

void SLICImgSeg(
  int* maskBuffer, float4* floatBuffer, 
  int nWidth, int nHeight, int nSegs, int nIt,
  SLICClusterCenter* vSLICCenterList, float weight);

#endif
