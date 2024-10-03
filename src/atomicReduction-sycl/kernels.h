#include <sycl/sycl.hpp>

inline int atomicAdd(int& val, const int delta)
{
  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(val);
  return ref.fetch_add(delta);
}

void atomic_reduction(int *in, int *out, int arrayLength, const sycl::nd_item<1> &item) {
  int sum = 0;
  int idx = item.get_global_id(0);
  for(int i= idx;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)) {
    sum+=in[i];
  }
  atomicAdd(out[0],sum);
}

void atomic_reduction_v2(int *in, int* out, int arrayLength, const sycl::nd_item<1> &item) {
  int sum = 0;
  int idx = item.get_global_id(0);
  for(int i= idx*2;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*2) {
    sum+=in[i] + in[i+1];
  }
  atomicAdd(out[0],sum);
}

void atomic_reduction_v4(int *in, int* out, int arrayLength, const sycl::nd_item<1> &item) {
  int sum = 0;
  int idx = item.get_global_id(0);
  for(int i= idx*4;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*4) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3];
  }
  atomicAdd(out[0],sum);
}

void atomic_reduction_v8(int *in, int* out, int arrayLength, const sycl::nd_item<1> &item) {
  int sum = 0;
  int idx = item.get_global_id(0);
  for(int i= idx*8;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*8) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +
      in[i+4] + in[i+5] + in[i+6] + in[i+7];
  }
  atomicAdd(out[0],sum);
}

void atomic_reduction_v16(int *in, int* out, int arrayLength, const sycl::nd_item<1> &item) {
  int sum = 0;
  int idx = item.get_global_id(0);
  for(int i= idx*16;i<arrayLength;i+=item.get_local_range(0)*item.get_group_range(0)*16) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +
      in[i+4] + in[i+5] + in[i+6] + in[i+7] +
      in[i+8] + in[i+9] + in[i+10] + in[i+11] +
      in[i+12] +in[i+13] + in[i+14] + in[i+15] ;
  }
  atomicAdd(out[0],sum);
}

