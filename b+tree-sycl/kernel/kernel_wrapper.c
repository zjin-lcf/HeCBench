#include <string.h>
#include <stdio.h>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel_wrapper.h"


void 
kernel_wrapper(	
    queue &q,
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

  long long offload_start = get_time();

  { // SYCL scope

    const property_list props = property::buffer::use_host_ptr();

    buffer<record,1> recordsD (records, records_mem, props);
    buffer<knode,1> knodesD (knodes, knodes_mem, props);
    buffer<long,1> currKnodeD (currKnode, count, props);
    buffer<long,1> offsetD (offset, count, props);
    buffer<int,1> keysD (keys, count, props);
    buffer<record,1> ansD (ans, count, props);
    currKnodeD.set_final_data(nullptr);
    offsetD.set_final_data(nullptr);


    // findK kernel

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
        auto recordsD_acc = recordsD.get_access<sycl_read>(cgh);
        auto offsetD_acc = offsetD.get_access<sycl_read_write>(cgh);
        auto keysD_acc = keysD.get_access<sycl_read>(cgh);
        auto ansD_acc = ansD.get_access<sycl_write>(cgh);

        cgh.parallel_for<class findK>(
            nd_range<1>(range<1>(global_work_size[0]),
              range<1>(local_work_size[0])), [=] (nd_item<1> item) {
#include "findK.sycl"
            });
        });

    q.wait();
  } // SYCL scope
  long long offload_end = get_time();

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("ans[%d] = %d\n", i, ans[i].value);
  printf("\n");
#endif
  printf("Device offloading time:\n");
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000);
}

