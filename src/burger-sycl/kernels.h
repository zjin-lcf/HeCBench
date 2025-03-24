#define idx(i,j)   (i)*x_points+(j)

void core (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const double *__restrict__ u,
    const double *__restrict__ v,
    const int x_points,
    const int y_points,
    const double nu,
    const double del_t,
    const double del_x,
    const double del_y)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int i = item.get_global_id(1) + 1;
      int j = item.get_global_id(2) + 1;
      if (j < x_points - 1 && i < y_points - 1) {
        u_new[idx(i,j)] = u[idx(i,j)] +
          (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) +
          (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) -
          (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) -
          (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

        v_new[idx(i,j)] = v[idx(i,j)] +
          (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) +
          (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
          (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) -
          (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void bound_h (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int i = item.get_global_id(2);
      if (i < x_points) {
        u_new[idx(0,i)] = 1.0;
        v_new[idx(0,i)] = 1.0;
        u_new[idx(y_points-1,i)] = 1.0;
        v_new[idx(y_points-1,i)] = 1.0;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void bound_v (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int j = item.get_global_id(2);
      if (j < y_points) {
        u_new[idx(j,0)] = 1.0;
        v_new[idx(j,0)] = 1.0;
        u_new[idx(j,x_points-1)] = 1.0;
        v_new[idx(j,x_points-1)] = 1.0;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void update (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    double *__restrict__ u,
    double *__restrict__ v,
    const double *__restrict__ u_new,
    const double *__restrict__ v_new,
    const int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int i = item.get_global_id(2);
      if (i < n) {
        u[i] = u_new[i];
        v[i] = v_new[i];
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
