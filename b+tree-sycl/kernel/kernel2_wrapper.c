#include <string.h>
#include <stdio.h>
#include "b+tree.h"
#include "timer.h"
#include "kernel2_wrapper.h"

void 
kernel2_wrapper(
    queue &q,
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

  long long offload_start = get_time();

  { // SYCL scope

    const property_list props = property::buffer::use_host_ptr();

    buffer<knode,1> knodesD (knodes, knodes_mem, props);
    buffer<long,1> currKnodeD (currKnode, count, props);
    buffer<long,1> offsetD (offset, count, props);
    buffer<long,1> lastKnodeD (lastKnode, count, props);
    buffer<long,1> offset_2D (offset_2, count, props);
    buffer<int,1> startD (start, count, props);
    buffer<int,1> endD (end, count, props);
    buffer<int,1> ansDStart (recstart, count, props);
    buffer<int,1> ansDLength (reclength, count, props);
    currKnodeD.set_final_data(nullptr);
    lastKnodeD.set_final_data(nullptr);
    offsetD.set_final_data(nullptr);
    offset_2D.set_final_data(nullptr);

    // findRangeK kernel

    size_t local_work_size[1];
#ifdef USE_GPU
    local_work_size[0] = order < 256 ? order : 256;
#else
    local_work_size[0] = order < 1024 ? order : 1024;
#endif
    size_t global_work_size[1];
    global_work_size[0] = count * local_work_size[0];

    printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

    q.submit([&](handler& cgh) {

        auto knodesD_acc = knodesD.get_access<sycl_read>(cgh);
        auto currKnodeD_acc = currKnodeD.get_access<sycl_read_write>(cgh);
        auto lastKnodeD_acc = lastKnodeD.get_access<sycl_read_write>(cgh);
        auto offsetD_acc = offsetD.get_access<sycl_read_write>(cgh);
        auto offset_2D_acc = offset_2D.get_access<sycl_read_write>(cgh);
        auto startD_acc = startD.get_access<sycl_read>(cgh);
        auto endD_acc = endD.get_access<sycl_read>(cgh);
        auto RecstartD_acc = ansDStart.get_access<sycl_read_write>(cgh);
        auto ReclenD_acc = ansDLength.get_access<sycl_write>(cgh);

        cgh.parallel_for<class findRangeK>(
            nd_range<1>(range<1>(global_work_size[0]),
              range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "findRangeK.sycl"
            });
        });
    q.wait();
  } // SYCL scope
  long long offload_end = get_time();

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("recstart[%d] = %d\n", i, recstart[i]);
  for (int i = 0; i < count; i++)
    printf("reclength[%d] = %d\n", i, reclength[i]);
#endif

  printf("Device offloading time:\n");
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000);

}

