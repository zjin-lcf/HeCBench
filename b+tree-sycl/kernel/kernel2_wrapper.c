#include <string.h>
#include <stdio.h>
#include "b+tree.h"
#include "timer.h"
#include "kernel2_wrapper.h"

void 
kernel2_wrapper(
    sycl::queue &q,
    knode *knodes,
    long knodes_elem,
    long knodes_mem,  // not length in byte

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
  knode *knodesD_acc = sycl::malloc_device<knode>(knodes_mem, q);
  q.memcpy(knodesD_acc, knodes, sizeof(knode) * knodes_mem);

  long *currKnodeD_acc = sycl::malloc_device<long>(count, q); 
  q.memcpy(currKnodeD_acc, currKnode, sizeof(long) * count);

  long *offsetD_acc = sycl::malloc_device<long>(count, q);
  q.memcpy(offsetD_acc, offset, sizeof(long) * count); 

  long *lastKnodeD_acc = sycl::malloc_device<long>(count, q);
  q.memcpy(lastKnodeD_acc, lastKnode, sizeof(long) * count);

  long *offset_2D_acc = sycl::malloc_device<long>(count, q);
  q.memcpy(offset_2D_acc, offset_2, sizeof(long) * count);

  int *startD_acc = sycl::malloc_device<int>(count, q);
  q.memcpy(startD_acc, start, sizeof(int) * count);

  int *endD_acc = sycl::malloc_device<int>(count, q);
  q.memcpy(endD_acc, end, sizeof(int) * count);

  int *RecstartD_acc = sycl::malloc_device<int>(count, q);
  q.memcpy(RecstartD_acc, recstart, sizeof(int) * count);

  int *ReclenD_acc = sycl::malloc_device<int>(count, q);

  size_t local_work_size[1];
#ifdef USE_GPU
  local_work_size[0] = order < 256 ? order : 256;
#else
  local_work_size[0] = order < 1024 ? order : 1024;
#endif
  size_t global_work_size[1];
  global_work_size[0] = count * local_work_size[0];

#ifdef DEBUG
  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n",
         (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);
#endif

  q.wait();
  long long kernel_start = get_time();

  // findRangeK kernel
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class findRangeK>(sycl::nd_range<1>(
      sycl::range<1>(global_work_size[0]),
      sycl::range<1>(local_work_size[0])), [=] (sycl::nd_item<1> item) {
      #include "findRangeK.sycl"
    });
  }).wait();

  long long kernel_end = get_time();
  printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));

  q.memcpy(recstart, RecstartD_acc, count*sizeof(int));
  q.memcpy(reclength, ReclenD_acc, count*sizeof(int));
  q.wait();

  sycl::free(knodesD_acc, q);
  sycl::free(currKnodeD_acc, q);
  sycl::free(offsetD_acc, q);
  sycl::free(lastKnodeD_acc, q);
  sycl::free(offset_2D_acc, q);
  sycl::free(startD_acc, q);
  sycl::free(endD_acc, q);
  sycl::free(RecstartD_acc, q);
  sycl::free(ReclenD_acc, q);

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("recstart[%d] = %d\n", i, recstart[i]);
  for (int i = 0; i < count; i++)
    printf("reclength[%d] = %d\n", i, reclength[i]);
#endif
}

