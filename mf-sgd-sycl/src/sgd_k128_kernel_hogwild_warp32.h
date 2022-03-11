void sgd_k128_kernel_hogwild_warp32_lrate(
    const mf_node *__restrict R, long long nnz, sycl::half *__restrict p,
    sycl::half *__restrict q, unsigned int *state,
    const float *__restrict dynamic_rate, long long u_seg, long long v_seg,
    int k, int num_iters, int current_iter, int update_count_per_block,
    int update_count_this_block, int update_vector_size, float lambda_p,
    float lambda_q, int u_grid, int v_grid, int u_id, int v_id,
    sycl::nd_item<1> &item)
{
  //persistant thread
  for(int ite = current_iter; ite < current_iter + num_iters; ite ++)
  {
    float tmp_lrate = dynamic_rate[ite];

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
      int lane_id = item.get_local_id(0) % 32;
      int local_wid = item.get_local_id(0) / 32;
      int wid = 4 * item.get_group(0) + local_wid;

      long long start_id = 0;
      if(lane_id == 0)
      {
        long long origin = (long long)(LCG_random(state+wid)*nnz);
        start_id = origin%nnz;
      }
      auto sg = item.get_sub_group();
      start_id = sycl::select_from_group(sg, start_id, 0);

      for(int i = 0;i < update_vector_size;i++)
      {
        int offset = (start_id + i)%nnz;

        float r = R[offset].rate;
        int u = R[offset].u;
        int v = R[offset].v;

        //read the p & q into register file.
        int base_p = u*k;
        int base_q = v*k;

        float tmp_p1 = sycl::vec<sycl::half, 1>{p[base_p + lane_id]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];
        float tmp_q1 = sycl::vec<sycl::half, 1>{q[base_q + lane_id]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];

        float tmp_p2 = sycl::vec<sycl::half, 1>{p[base_p + lane_id + 32]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];
        float tmp_q2 = sycl::vec<sycl::half, 1>{q[base_q + lane_id + 32]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];

        float tmp_p3 = sycl::vec<sycl::half, 1>{p[base_p + lane_id + 64]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];
        float tmp_q3 = sycl::vec<sycl::half, 1>{q[base_q + lane_id + 64]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];

        float tmp_p4 = sycl::vec<sycl::half, 1>{p[base_p + lane_id + 96]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];
        float tmp_q4 = sycl::vec<sycl::half, 1>{q[base_q + lane_id + 96]}
                           .convert<float, sycl::rounding_mode::automatic>()[0];

        float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

        //get dot product.
        tmp_product += sycl::shift_group_left(sg, tmp_product, 16);
        tmp_product += sycl::shift_group_left(sg, tmp_product, 8);
        tmp_product += sycl::shift_group_left(sg, tmp_product, 4);
        tmp_product += sycl::shift_group_left(sg, tmp_product, 2);
        tmp_product += sycl::shift_group_left(sg, tmp_product, 1);
        tmp_product = sycl::select_from_group(sg, tmp_product, 0);

        float ruv = r - tmp_product;

        //update
        //only works for k=blockDim.x=128
        p[base_p + lane_id + 0] =
            sycl::vec<float, 1>{tmp_p1 +
                                tmp_lrate * (ruv * tmp_q1 - lambda_p * tmp_p1)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        q[base_q + lane_id + 0] =
            sycl::vec<float, 1>{tmp_q1 +
                                tmp_lrate * (ruv * tmp_p1 - lambda_q * tmp_q1)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];

        p[base_p + lane_id + 32] =
            sycl::vec<float, 1>{tmp_p2 +
                                tmp_lrate * (ruv * tmp_q2 - lambda_p * tmp_p2)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        q[base_q + lane_id + 32] =
            sycl::vec<float, 1>{tmp_q2 +
                                tmp_lrate * (ruv * tmp_p2 - lambda_q * tmp_q2)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];

        p[base_p + lane_id + 64] =
            sycl::vec<float, 1>{tmp_p3 +
                                tmp_lrate * (ruv * tmp_q3 - lambda_p * tmp_p3)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        q[base_q + lane_id + 64] =
            sycl::vec<float, 1>{tmp_q3 +
                                tmp_lrate * (ruv * tmp_p3 - lambda_q * tmp_q3)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];

        p[base_p + lane_id + 96] =
            sycl::vec<float, 1>{tmp_p4 +
                                tmp_lrate * (ruv * tmp_q4 - lambda_p * tmp_p4)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        q[base_q + lane_id + 96] =
            sycl::vec<float, 1>{tmp_q4 +
                                tmp_lrate * (ruv * tmp_p4 - lambda_q * tmp_q4)}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
      }    
    }
  }
}



