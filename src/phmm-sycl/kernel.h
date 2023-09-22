void pair_HMM_forward(
    sycl::nd_item<1> &item,
    const int cur_i,
    const int cur_j,
    const fArray *__restrict forward_matrix_in,
    const tArray *__restrict transitions,
    const fArray *__restrict emissions,
    const lArray *__restrict likelihood,
    const sArray *__restrict start_transitions,
          fArray *__restrict forward_matrix_out)
{
  auto g = item.get_group();

  sycl::multi_ptr<double[batch][states-1], sycl::access::address_space::local_space>
  p1 = sycl::ext::oneapi::group_local_memory_for_overwrite<double[batch][states-1]>(g);
  double(*e)[states-1] = *p1;

  sycl::multi_ptr<double[1][batch][2], sycl::access::address_space::local_space>
  p2 = sycl::ext::oneapi::group_local_memory_for_overwrite<double[1][batch][2]>(g);
  double(*f01)[batch][2] = *p2;

  sycl::multi_ptr<double[1][batch][2], sycl::access::address_space::local_space>
  p3 = sycl::ext::oneapi::group_local_memory_for_overwrite<double[1][batch][2]>(g);
  double(*mul_3d)[batch][2] = *p3;

  sycl::multi_ptr<double[4][batch][1][2], sycl::access::address_space::local_space>
  p4 = sycl::ext::oneapi::group_local_memory_for_overwrite<double[4][batch][1][2]>(g);
  double(*mul_4d)[batch][1][2] = *p4;

  int batch_id = item.get_group(0);
  int states_id = item.get_local_id(0);

  e[batch_id][states_id] = emissions[cur_i][cur_j][batch_id][states_id];

  double t[2][2][batch][2][2];
  for (int k = 0; k < 2; k++) {
    for (int l = 0; l < 2; l++) {
      t[0][0][batch_id][k][l] = transitions[cur_i - 1][batch_id][k][l];
      t[0][1][batch_id][k][l] = transitions[cur_i - 1][batch_id][k][l];
      t[1][0][batch_id][k][l] = transitions[cur_i][batch_id][k][l];
      t[1][1][batch_id][k][l] = transitions[cur_i][batch_id][k][l];
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (cur_i > 0 && cur_j == 0) {
    if (cur_i == 1) {
      forward_matrix_out[1][0][batch_id][states_id] = 
        start_transitions[batch_id][states_id] * e[0][states_id];
    }
    else {
      double t01[batch][2][2];
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          t01[batch_id][j][k] = t[0][1][batch_id][j][k];
        }
      }

      f01[0][batch_id][states_id] = 
        forward_matrix_in[cur_i - 1][cur_j][batch_id][states_id];

      item.barrier(sycl::access::fence_space::local_space);

      double s = 0.0;
      for (int k = 0; k < 2; k++)
        s += f01[0][batch_id][k] * t01[batch_id][k][states_id];
      s *= (e[batch_id][states_id] * likelihood[0][1][batch_id][states_id]);
      mul_3d[0][batch_id][states_id] = s;

      item.barrier(sycl::access::fence_space::local_space);

      forward_matrix_out[cur_i][0][batch_id][states_id] = mul_3d[0][batch_id][states_id];
    }
  }
  else if (cur_i > 0 and cur_j > 0) {

    double f[2][2][batch][1][2];
    for (int i = 0; i < 2; i++) {
      f[0][0][batch_id][0][i] = forward_matrix_in[cur_i-1][cur_j-1][batch_id][i];
      f[0][1][batch_id][0][i] = forward_matrix_in[cur_i-1][cur_j][batch_id][i];
      f[1][0][batch_id][0][i] = forward_matrix_in[cur_i][cur_j-1][batch_id][i];
      f[1][1][batch_id][0][i] = forward_matrix_in[cur_i][cur_j][batch_id][i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;

    for (int k = 0; k < 2; k++) {
      s0 += f[0][0][batch_id][0][k] * t[0][0][batch_id][k][states_id];
      s1 += f[0][1][batch_id][0][k] * t[0][1][batch_id][k][states_id];
      s2 += f[1][0][batch_id][0][k] * t[1][0][batch_id][k][states_id];
      s3 += f[1][1][batch_id][0][k] * t[1][1][batch_id][k][states_id];
    }
    s0 *= likelihood[0][0][batch_id][states_id];
    s1 *= likelihood[0][1][batch_id][states_id];
    s2 *= likelihood[1][0][batch_id][states_id];
    s3 *= likelihood[1][1][batch_id][states_id];
    mul_4d[0][batch_id][0][states_id] = s0;
    mul_4d[1][batch_id][0][states_id] = s1;
    mul_4d[2][batch_id][0][states_id] = s2;
    mul_4d[3][batch_id][0][states_id] = s3;

    item.barrier(sycl::access::fence_space::local_space);

    for (int j = 0; j < 2; j++) {
      double summation = mul_4d[0][batch_id][0][j] + 
                         mul_4d[1][batch_id][0][j] +
                         mul_4d[2][batch_id][0][j] +
                         mul_4d[3][batch_id][0][j];

      summation *= e[batch_id][j];

      forward_matrix_out[cur_i][cur_j][batch_id][j] = summation;
    }
  }
}
