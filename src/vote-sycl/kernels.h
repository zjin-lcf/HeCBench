#ifndef SIMPLEVOTE_KERNEL_CU
#define SIMPLEVOTE_KERNEL_CU

////////////////////////////////////////////////////////////////////////////////
// Vote Any/All intrinsic kernel function tests are supported only by CUDA
// capable devices that are CUDA hardware that has SM1.2 or later
// Vote Functions (refer to section 4.4.5 in the CUDA Programming Guide)
////////////////////////////////////////////////////////////////////////////////

// Kernel #1 tests the across-the-warp vote(any) intrinsic.
// If ANY one of the threads (within the warp) of the predicated condition
// returns a non-zero value, then all threads within this warp will return a
// non-zero value
void VoteAnyKernel1(const unsigned int *input, unsigned int *result,
                    int repeat, sycl::nd_item<1> &item) {
  int tx = item.get_local_id(0);
  auto sg = item.get_sub_group();
  for (int i = 0; i < repeat; i++)
    result[tx] = sycl::any_of_group(sg, input[tx]);
}

// Kernel #2 tests the across-the-warp vote(all) intrinsic.
// If ALL of the threads (within the warp) of the predicated condition returns
// a non-zero value, then all threads within this warp will return a non-zero
// value
void VoteAllKernel2(const unsigned int *input, unsigned int *result,
                    int repeat, sycl::nd_item<1> &item) {
  int tx = item.get_local_id(0);
  auto sg = item.get_sub_group();
  for (int i = 0; i < repeat; i++)
    result[tx] = sycl::all_of_group(sg, input[tx]);
}

// Kernel #3 is a directed test for the across-the-warp vote(all) intrinsic.
// This kernel will test for conditions across warps, and within half warps
void VoteAnyKernel3(bool *info, int warp_size, int repeat,
                    sycl::nd_item<1> &item) {
  int tx = item.get_local_id(0);
  auto sg = item.get_sub_group();
  for (int i = 0; i < repeat; i++) {
    bool *offs = info + (tx * 3);

    // The following should hold true for the second and third warp
    *offs = sycl::any_of_group(sg, (tx >= (warp_size * 3) / 2));

    // The following should hold true for the "upper half" of the second warp,
    // and all of the third warp
    *(offs + 1) = (tx >= (warp_size * 3) / 2 ? true : false);

    // The following should hold true for the third warp only
    if (sycl::all_of_group(sg, (tx >= (warp_size * 3) / 2))) {
      *(offs + 2) = true;
    }
  }
}

#endif
