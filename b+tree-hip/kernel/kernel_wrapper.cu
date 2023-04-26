#include <hip/hip_runtime.h>
#include <stdio.h>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel.cu"
#include "./kernel_wrapper.h"        // (in current directory)

void 
kernel_wrapper(record *records,
    long records_mem,
    knode *knodes,
    long knodes_elem,
    long knodes_mem,

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    int *keys,
    record *ans)
{

  int numBlocks;
  numBlocks = count;                  // max # of blocks can be 65,535
  int threadsPerBlock;
  threadsPerBlock = order < 256 ? order : 256;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

  //==================================================50
  //  recordsD
  //==================================================50

  record *recordsD;
  hipMalloc((void**)&recordsD, records_mem);

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
  //  keysD
  //==================================================50

  int *keysD;
  hipMalloc((void**)&keysD, count*sizeof(int));

  //==================================================50
  //  ansD
  //==================================================50

  record *ansD;
  hipMalloc((void**)&ansD, count*sizeof(record));

  //==================================================50
  //  recordsD
  //==================================================50

  hipMemcpyAsync(recordsD, records, records_mem, hipMemcpyHostToDevice, 0);

  //==================================================50
  //  knodesD
  //==================================================50

  hipMemcpyAsync(knodesD, knodes, knodes_mem, hipMemcpyHostToDevice, 0);

  //==================================================50
  //  currKnodeD
  //==================================================50

  hipMemcpyAsync(currKnodeD, currKnode, count*sizeof(long), hipMemcpyHostToDevice, 0);

  //==================================================50
  //  offsetD
  //==================================================50

  hipMemcpyAsync(offsetD, offset, count*sizeof(long), hipMemcpyHostToDevice, 0);

  //==================================================50
  //  keysD
  //==================================================50

  hipMemcpyAsync(keysD, keys, count*sizeof(int), hipMemcpyHostToDevice, 0);

  //==================================================50
  //  ansD
  //==================================================50

  hipMemcpyAsync(ansD, ans, count*sizeof(record), hipMemcpyHostToDevice, 0);

  hipDeviceSynchronize();
  long long kernel_start = get_time();

  findK<<<numBlocks, threadsPerBlock>>>(  maxheight,
      knodesD,
      knodes_elem,
      recordsD,
      currKnodeD,
      offsetD,
      keysD,
      ansD);

  hipDeviceSynchronize();
  long long kernel_end = get_time();
  printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));

  hipMemcpy(ans, ansD, count*sizeof(record), hipMemcpyDeviceToHost);

  hipFree(recordsD);
  hipFree(knodesD);
  hipFree(currKnodeD);
  hipFree(offsetD);
  hipFree(keysD);
  hipFree(ansD);

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("ans[%d] = %d\n", i, ans[i].value);
  printf("\n");
#endif
}
