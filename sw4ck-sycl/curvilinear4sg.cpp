#include "curvilinear4sg.h"
#include "kernel1.cpp"
#include "kernel2.cpp"
#include "kernel3.cpp"
#include "kernel4.cpp"
#include "kernel5.cpp"

void curvilinear4sg_ci(
    queue &q,
    int ifirst, int ilast, 
    int jfirst, int jlast, 
    int kfirst, int klast,
    buffer<float_sw4,1> &d_u, 
    buffer<float_sw4,1> &d_mu,
    buffer<float_sw4,1> &d_lambda,
    buffer<float_sw4,1> &d_met,
    buffer<float_sw4,1> &d_jac,
    buffer<float_sw4,1> &d_lu, 
    int* onesided,
    buffer<float_sw4,1> &d_cof, 
    buffer<float_sw4,1> &d_sg_str, 
    int nk, char op) {

  float_sw4 a1 = 0;
  float_sw4 sgn = 1;
  if (op == '=') {
    a1 = 0;
    sgn = 1;
  } else if (op == '+') {
    a1 = 1;
    sgn = 1;
  } else if (op == '-') {
    a1 = 1;
    sgn = -1;
  }

  int kstart = kfirst + 2;
  int kend = klast - 2;
  if (onesided[5] == 1) kend = nk - 6;

  if (onesided[4] == 1) {
    kstart = 7;

    Range<16> I(ifirst + 2, ilast - 1);
    Range<4> J(jfirst + 2, jlast - 1);
    Range<3> K(1, 6 + 1);  // This was 6

    range<3> gws (K.tpb * K.blocks, J.tpb * J.blocks, I.tpb * I.blocks);
    range<3> lws (K.tpb, J.tpb, I.tpb);
    id<3> offset (K.start, J.start, I.start);

    q.submit([&] (handler &cgh) {
      auto u = d_u.get_access<sycl_read>(cgh);
      auto mu = d_mu.get_access<sycl_read>(cgh); 
      auto lambda = d_lambda.get_access<sycl_read>(cgh);
      auto met = d_met.get_access<sycl_read>(cgh);
      auto jac = d_jac.get_access<sycl_read>(cgh);
      auto lu = d_lu.get_access<sycl_read_write>(cgh); 
      auto cof = d_cof.get_access<sycl_read>(cgh); 
      auto str = d_sg_str.get_access<sycl_read>(cgh); 
      cgh.parallel_for<class k1>(nd_range<3>(gws, lws, offset), [=] (nd_item<3> item) {
        kernel1(
          item,
          I.end, J.end, K.end,
          ifirst, ilast, jfirst, jlast, kfirst, klast, a1, sgn,
          u.get_pointer(), 
          mu.get_pointer(), 
          lambda.get_pointer(),
          met.get_pointer(),
          jac.get_pointer(),
          lu.get_pointer(), 
          // acof, 
          cof.get_pointer() + 6,
          // bope, 
          cof.get_pointer() + 6 + 384 + 24,
          // ghcof, 
          cof.get_pointer() + 6 + 384 + 24 + 48,
          // acof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6,
          // ghcof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6 + 384,
          // strx
          str.get_pointer(), 
          // stry
          str.get_pointer() + ilast - ifirst + 1);
       });
     });
  }

  Range<64> I(ifirst + 2, ilast - 1);
  Range<2> J(jfirst + 2, jlast - 1);
  Range<2> K(kstart, kend + 1);  // Changed for CUrvi-MR Was klast-1

  range<3> gws (K.tpb * K.blocks, J.tpb * J.blocks, I.tpb * I.blocks);
  range<3> lws (K.tpb, J.tpb, I.tpb);
  id<3> offset (K.start, J.start, I.start);

  q.submit([&] (handler &cgh) {
    auto u = d_u.get_access<sycl_read>(cgh);
    auto mu = d_mu.get_access<sycl_read>(cgh); 
    auto lambda = d_lambda.get_access<sycl_read>(cgh);
    auto met = d_met.get_access<sycl_read>(cgh);
    auto jac = d_jac.get_access<sycl_read>(cgh);
    auto lu = d_lu.get_access<sycl_read_write>(cgh); 
    auto cof = d_cof.get_access<sycl_read>(cgh); 
    auto str = d_sg_str.get_access<sycl_read>(cgh); 
    cgh.parallel_for<class k2>(nd_range<3>(gws, lws, offset), [=] (nd_item<3> item) {
      kernel2(
          item,
          I.end, J.end, K.end,
          ifirst, ilast, jfirst, jlast, kfirst, klast, a1, sgn,
          u.get_pointer(), 
          mu.get_pointer(), 
          lambda.get_pointer(),
          met.get_pointer(),
          jac.get_pointer(),
          lu.get_pointer(), 
          // acof, 
          cof.get_pointer() + 6,
          // bope, 
          cof.get_pointer() + 6 + 384 + 24,
          // ghcof, 
          cof.get_pointer() + 6 + 384 + 24 + 48,
          // acof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6,
          // ghcof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6 + 384,
          // strx
          str.get_pointer(), 
          // stry
          str.get_pointer() + ilast - ifirst + 1);
    });
  });

  q.submit([&] (handler &cgh) {
    auto u = d_u.get_access<sycl_read>(cgh);
    auto mu = d_mu.get_access<sycl_read>(cgh); 
    auto lambda = d_lambda.get_access<sycl_read>(cgh);
    auto met = d_met.get_access<sycl_read>(cgh);
    auto jac = d_jac.get_access<sycl_read>(cgh);
    auto lu = d_lu.get_access<sycl_read_write>(cgh); 
    auto cof = d_cof.get_access<sycl_read>(cgh); 
    auto str = d_sg_str.get_access<sycl_read>(cgh); 
    cgh.parallel_for<class k3>(nd_range<3>(gws, lws, offset), [=] (nd_item<3> item) {
      kernel3(
          item,
          I.end, J.end, K.end,
          ifirst, ilast, jfirst, jlast, kfirst, klast, a1, sgn,
          u.get_pointer(), 
          mu.get_pointer(), 
          lambda.get_pointer(),
          met.get_pointer(),
          jac.get_pointer(),
          lu.get_pointer(), 
          // acof, 
          cof.get_pointer() + 6,
          // bope, 
          cof.get_pointer() + 6 + 384 + 24,
          // ghcof, 
          cof.get_pointer() + 6 + 384 + 24 + 48,
          // acof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6,
          // ghcof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6 + 384,
          // strx
          str.get_pointer(), 
          // stry
          str.get_pointer() + ilast - ifirst + 1);
    });
  });

  q.submit([&] (handler &cgh) {
    auto u = d_u.get_access<sycl_read>(cgh);
    auto mu = d_mu.get_access<sycl_read>(cgh); 
    auto lambda = d_lambda.get_access<sycl_read>(cgh);
    auto met = d_met.get_access<sycl_read>(cgh);
    auto jac = d_jac.get_access<sycl_read>(cgh);
    auto lu = d_lu.get_access<sycl_read_write>(cgh); 
    auto cof = d_cof.get_access<sycl_read>(cgh); 
    auto str = d_sg_str.get_access<sycl_read>(cgh); 
    cgh.parallel_for<class k4>(nd_range<3>(gws, lws, offset), [=] (nd_item<3> item) {
      kernel4(
          item,
          I.end, J.end, K.end,
          ifirst, ilast, jfirst, jlast, kfirst, klast, a1, sgn,
          u.get_pointer(), 
          mu.get_pointer(), 
          lambda.get_pointer(),
          met.get_pointer(),
          jac.get_pointer(),
          lu.get_pointer(), 
          // acof, 
          cof.get_pointer() + 6,
          // bope, 
          cof.get_pointer() + 6 + 384 + 24,
          // ghcof, 
          cof.get_pointer() + 6 + 384 + 24 + 48,
          // acof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6,
          // ghcof_no_gp, 
          cof.get_pointer() + 6 + 384 + 24 + 48 + 6 + 384,
          // strx
          str.get_pointer(), 
          // stry
          str.get_pointer() + ilast - ifirst + 1);
    });
  });

  if (onesided[5] == 1) {
    Range<16> I(ifirst + 2, ilast - 1);
    Range<4> J(jfirst + 2, jlast - 1);
    Range<4> K(nk - 5, nk + 1);  // THIS WAS 6

    range<3> gws (K.tpb * K.blocks, J.tpb * J.blocks, I.tpb * I.blocks);
    range<3> lws (K.tpb, J.tpb, I.tpb);
    id<3> offset (K.start, J.start, I.start);

    q.submit([&] (handler &cgh) {
      auto u = d_u.get_access<sycl_read>(cgh);
      auto mu = d_mu.get_access<sycl_read>(cgh); 
      auto lambda = d_lambda.get_access<sycl_read>(cgh);
      auto met = d_met.get_access<sycl_read>(cgh);
      auto jac = d_jac.get_access<sycl_read>(cgh);
      auto lu = d_lu.get_access<sycl_read_write>(cgh); 
      auto cof = d_cof.get_access<sycl_read>(cgh); 
      auto str = d_sg_str.get_access<sycl_read>(cgh); 
      cgh.parallel_for<class k5>(nd_range<3>(gws, lws, offset), [=] (nd_item<3> item) {
        kernel5(
            item,
            I.end, J.end, K.end,
            ifirst, ilast, jfirst, jlast, kfirst, klast, nk, a1, sgn, 
            u.get_pointer(), 
            mu.get_pointer(), 
            lambda.get_pointer(),
            met.get_pointer(),
            jac.get_pointer(),
            lu.get_pointer(), 
            // acof, 
            cof.get_pointer() + 6,
            // bope, 
            cof.get_pointer() + 6 + 384 + 24,
            // ghcof, 
            cof.get_pointer() + 6 + 384 + 24 + 48,
            // acof_no_gp, 
            cof.get_pointer() + 6 + 384 + 24 + 48 + 6,
            // ghcof_no_gp, 
            cof.get_pointer() + 6 + 384 + 24 + 48 + 6 + 384,
            // strx
            str.get_pointer(), 
            // stry
            str.get_pointer() + ilast - ifirst + 1);
       });
    });
  }
}


