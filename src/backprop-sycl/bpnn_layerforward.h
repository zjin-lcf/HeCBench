// begin of kernel_layerforward
void kernel_layerforward(
  sycl::queue &q,
  sycl::range<3> &gws,
  sycl::range<3> &lws,
  const int slm_size,
  const float*__restrict__ input,
        float*__restrict__ input_weights,
        float*__restrict__ hidden_partial_sum,
  const int hid)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1> input_node (sycl::range<1>(HEIGHT), cgh);
    sycl::local_accessor<float, 1> weight_matrix (sycl::range<1>(HEIGHT * WIDTH), cgh);

    auto kfn = [=] (sycl::nd_item<3> item) {
      int by = item.get_group(1);
      int tx = item.get_local_id(2);
      int ty = item.get_local_id(1);

      int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;

      int index_in = HEIGHT * by + ty + 1;

      if ( tx == 0 ) input_node[ty] = input[index_in] ;
      item.barrier(sycl::access::fence_space::local_space);

      weight_matrix[ty * WIDTH + tx] = input_weights[index];
      item.barrier(sycl::access::fence_space::local_space);

      weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];
      item.barrier(sycl::access::fence_space::local_space);

      for ( int i = 1 ; i <= HEIGHT ; i=i*2){
        int power_two = i;

        if( ty % power_two == 0 )
          weight_matrix[ty * WIDTH + tx] += weight_matrix[(ty + power_two/2)* WIDTH + tx];

        item.barrier(sycl::access::fence_space::local_space);

      }

      input_weights[index] =  weight_matrix[ty * WIDTH + tx];

      item.barrier(sycl::access::fence_space::local_space);

      if ( tx == 0 )
        hidden_partial_sum[by * hid + ty] = weight_matrix[tx* WIDTH + ty];
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
// end of kernel_layerforward
