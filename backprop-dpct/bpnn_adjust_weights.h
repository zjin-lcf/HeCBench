#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void kernel_adjust_weights(const float *ly, float *w, const float *delta,
                           float *oldw, const int hid,
                           sycl::nd_item<3> item_ct1)
{
  int by = item_ct1.get_group(1);
  int tx = item_ct1.get_local_id(2);
  int ty = item_ct1.get_local_id(1);

int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
int index_y = HEIGHT * by + ty + 1;
int index_x = tx + 1;

w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

  item_ct1.barrier(sycl::access::fence_space::local_space);

if (ty == 0 && by ==0){
  w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
}

}

