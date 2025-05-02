// begin of kernel_adjust_weights
void kernel_adjust_weights (
  sycl::queue &q,
  sycl::range<3> &gws,
  sycl::range<3> &lws,
  const int slm_size,
  const float*__restrict__ ly,
        float*__restrict__ w,
  const float*__restrict__ delta,
        float*__restrict__ oldw,
  const int hid)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {

      int by = item.get_group(1);
      int tx = item.get_local_id(2);
      int ty = item.get_local_id(1);

      int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;
      int index_y = HEIGHT * by + ty + 1;
      int index_x = tx + 1;

      w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

      item.barrier(sycl::access::fence_space::local_space);

      if (ty == 0 && by ==0){
        w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
      }

    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
// end of kernel_adjust_weights

