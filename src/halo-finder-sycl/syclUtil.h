#pragma once
#include <sycl/sycl.hpp>
#include <stdio.h>

inline void atomicWarpReduceAndUpdate(POSVEL_T *out, POSVEL_T val,
                                      sycl::nd_item<2> &item) {
  auto sg = item.get_sub_group();
  //perform shfl reduction
  val += sycl::shift_group_left(sg, val, 16);
  val += sycl::shift_group_left(sg, val, 8);
  val += sycl::shift_group_left(sg, val, 4);
  val += sycl::shift_group_left(sg, val, 2);
  val += sycl::shift_group_left(sg, val, 1);

  if (item.get_local_id(1) == 0) {
     auto aref = sycl::atomic_ref<POSVEL_T,
                 sycl::memory_order::relaxed,
                 sycl::memory_scope::device,
                 sycl::access::address_space::global_space> (out[0]);
     aref.fetch_add(val);
  }
}

class syclDeviceSelector {
  public:
  syclDeviceSelector() {
    char* str;
    int local_rank = 0;
    int numDev=1;

    //No MPI at this time so go by enviornment variables.
    //This may need to be updated to match your MPI flavor
    if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("SLURM_LOCALID")) != NULL) {
      local_rank = atoi(str);
    }

    //get the number of devices to use
    if((str = getenv("HACC_NUM_SYCL_DEV")) != NULL) {
      numDev=atoi(str);
    }

    int dev;
    //set via local MPI rank
    dev = local_rank % numDev;

    // TODO (syclSetDevice)
  }
};
