#include <string.h>
#include <stdio.h>
#include "b+tree.h"
#include "timer.h"
#include "kernel_wrapper.h"


void 
kernel_wrapper(	
    sycl::queue &q,
    record *records,
    long records_mem, // not length in byte
    knode *knodes,
    long knodes_elem,
    long knodes_mem,  // not length in byte

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    int *keys,
    record *ans)
{
  record *recordsD_acc = sycl::malloc_device<record>(records_mem, q);
  q.memcpy(recordsD_acc, records, sizeof(record) * records_mem);

  knode *knodesD_acc = sycl::malloc_device<knode>(knodes_mem, q);
  q.memcpy(knodesD_acc, knodes, sizeof(knode) * knodes_mem);

  long *currKnodeD_acc = sycl::malloc_device<long>(count, q);
  q.memcpy(currKnodeD_acc, currKnode, sizeof(long) * count);
 
  long *offsetD_acc = sycl::malloc_device<long>(count, q);
  q.memcpy(offsetD_acc, offset, sizeof(long) * count);

  int *keysD_acc = sycl::malloc_device<int>(count, q);
  q.memcpy(keysD_acc, keys, sizeof(int) * count);

  record *ansD_acc = sycl::malloc_device<record>(count, q);

  // findK kernel

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

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class findK>(sycl::nd_range<1>(
      sycl::range<1>(global_work_size[0]),
      sycl::range<1>(local_work_size[0])), [=] (sycl::nd_item<1> item) {
      #include "findK.sycl"
    });
  }).wait();

  long long kernel_end = get_time();

  q.memcpy(ans, ansD_acc, count*sizeof(record)).wait();

  sycl::free(recordsD_acc, q);
  sycl::free(knodesD_acc, q);
  sycl::free(currKnodeD_acc, q);
  sycl::free(offsetD_acc, q);
  sycl::free(keysD_acc, q);
  sycl::free(ansD_acc, q);

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("ans[%d] = %d\n", i, ans[i].value);
  printf("\n");
#endif

  printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));
}

