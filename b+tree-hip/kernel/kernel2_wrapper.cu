#include <stdio.h>
#include <hip/hip_runtime.h>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel2.cu"
#include "./kernel2_wrapper.h"

void 
kernel2_wrapper(  knode *knodes,
    long knodes_elem,
    long knodes_mem,

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    long *lastKnode,
    long *offset_2,
    int *start,
    int *end,
    int *recstart,
    int *reclength)
{

  long long offload_start = get_time();

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 256 ? order : 256;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);


  //==================================================50
  //  knodesD
  //==================================================50

  knode *knodesD;
  hipMalloc((void**)&knodesD, knodes_mem);

  //==================================================50
  //  currKnodeD
  //==================================================50

  long *currKnodeD;
  hipMalloc((void**)&currKnodeD, count*sizeof(long));

  //==================================================50
  //  offsetD
  //==================================================50

  long *offsetD;
  hipMalloc((void**)&offsetD, count*sizeof(long));

  //==================================================50
  //  lastKnodeD
  //==================================================50

  long *lastKnodeD;
  hipMalloc((void**)&lastKnodeD, count*sizeof(long));

  //==================================================50
  //  offset_2D
  //==================================================50

  long *offset_2D;
  hipMalloc((void**)&offset_2D, count*sizeof(long));

  //==================================================50
  //  startD
  //==================================================50

  int *startD;
  hipMalloc((void**)&startD, count*sizeof(int));

  //==================================================50
  //  endD
  //==================================================50

  int *endD;
  hipMalloc((void**)&endD, count*sizeof(int));

  //==================================================50
  //  ansDStart
  //==================================================50

  int *ansDStart;
  hipMalloc((void**)&ansDStart, count*sizeof(int));

  //==================================================50
  //  ansDLength
  //==================================================50

  int *ansDLength;
  hipMalloc((void**)&ansDLength, count*sizeof(int));

  hipMemcpyAsync(knodesD, knodes, knodes_mem, hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(currKnodeD, currKnode, count*sizeof(long), hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(offsetD, offset, count*sizeof(long), hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(lastKnodeD, lastKnode, count*sizeof(long), hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(offset_2D, offset_2, count*sizeof(long), hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(startD, start, count*sizeof(int), hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(endD, end, count*sizeof(int), hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(ansDStart, recstart, count*sizeof(int), hipMemcpyHostToDevice, 0);

  hipMemcpyAsync(ansDLength, reclength, count*sizeof(int), hipMemcpyHostToDevice, 0);

  // [GPU] findRangeK kernel
  hipLaunchKernelGGL(findRangeK, dim3(numBlocks), dim3(threadsPerBlock), 0, 0,   maxheight,
      knodesD,
      knodes_elem,
      currKnodeD,
      offsetD,
      lastKnodeD,
      offset_2D,
      startD,
      endD,
      ansDStart,
      ansDLength);

  hipMemcpyAsync(recstart, ansDStart, count*sizeof(int), hipMemcpyDeviceToHost, 0);

  hipMemcpyAsync(reclength, ansDLength, count*sizeof(int), hipMemcpyDeviceToHost, 0);

  hipDeviceSynchronize();

  hipFree(knodesD);
  hipFree(currKnodeD);
  hipFree(offsetD);
  hipFree(lastKnodeD);
  hipFree(offset_2D);
  hipFree(startD);
  hipFree(endD);
  hipFree(ansDStart);
  hipFree(ansDLength);

  long long offload_end = get_time();

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("recstart[%d] = %d\n", i, recstart[i]);
  for (int i = 0; i < count; i++)
    printf("reclength[%d] = %d\n", i, reclength[i]);
#endif


  printf("Total time:\n"); 
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000); 
}

