#include <sycl/sycl.hpp>
#include "curvilinear4sg.h"
#include "kernel1.cpp"
#include "kernel2.cpp"
#include "kernel3.cpp"
#include "kernel4.cpp"
#include "kernel5.cpp"

void curvilinear4sg_ci(sycl::queue &q,
                       int ifirst, int ilast, int jfirst, int jlast, int kfirst,
                       int klast, float_sw4 *d_u, float_sw4 *d_mu,
                       float_sw4 *d_lambda, float_sw4 *d_met, float_sw4 *d_jac,
                       float_sw4 *d_lu, int *onesided, float_sw4 *d_acof,
                       float_sw4 *d_bope, float_sw4 *d_ghcof,
                       float_sw4 *d_acof_no_gp, float_sw4 *d_ghcof_no_gp,
                       float_sw4 *d_strx, float_sw4 *d_stry, int nk, char op)
{
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

    sycl::range<3> tpb(K.tpb, J.tpb, I.tpb);
    sycl::range<3> blocks(K.blocks, J.blocks, I.blocks);

    q.parallel_for(
        sycl::nd_range<3>(blocks * tpb, tpb), [=](sycl::nd_item<3> item) {
          kernel1(I.start, I.end, J.start, J.end, K.start, K.end, ifirst, ilast,
                  jfirst, jlast, kfirst, klast, a1, sgn, d_u, d_mu, d_lambda,
                  d_met, d_jac, d_lu, d_acof, d_bope, d_ghcof, d_acof_no_gp,
                  d_ghcof_no_gp, d_strx, d_stry, item);
        });
  }

  Range<64> I(ifirst + 2, ilast - 1);
  Range<2> J(jfirst + 2, jlast - 1);
  Range<2> K(kstart, kend + 1);  // Changed for CUrvi-MR Was klast-1

  sycl::range<3> tpb(K.tpb, J.tpb, I.tpb);
  sycl::range<3> blocks(K.blocks, J.blocks, I.blocks);

  q.parallel_for(
      sycl::nd_range<3>(blocks * tpb, tpb), [=](sycl::nd_item<3> item) {
        kernel2(I.start, I.end, J.start, J.end, K.start, K.end, ifirst, ilast,
                jfirst, jlast, kfirst, klast, a1, sgn, d_u, d_mu, d_lambda,
                d_met, d_jac, d_lu, d_acof, d_bope, d_ghcof, d_acof_no_gp,
                d_ghcof_no_gp, d_strx, d_stry, item);
      });

  q.parallel_for(
      sycl::nd_range<3>(blocks * tpb, tpb), [=](sycl::nd_item<3> item) {
        kernel3(I.start, I.end, J.start, J.end, K.start, K.end, ifirst, ilast,
                jfirst, jlast, kfirst, klast, a1, sgn, d_u, d_mu, d_lambda,
                d_met, d_jac, d_lu, d_acof, d_bope, d_ghcof, d_acof_no_gp,
                d_ghcof_no_gp, d_strx, d_stry, item);
      });

  q.parallel_for(
      sycl::nd_range<3>(blocks * tpb, tpb), [=](sycl::nd_item<3> item) {
        kernel4(I.start, I.end, J.start, J.end, K.start, K.end, ifirst, ilast,
                jfirst, jlast, kfirst, klast, a1, sgn, d_u, d_mu, d_lambda,
                d_met, d_jac, d_lu, d_acof, d_bope, d_ghcof, d_acof_no_gp,
                d_ghcof_no_gp, d_strx, d_stry, item);
      });

  if (onesided[5] == 1) {
    Range<16> I(ifirst + 2, ilast - 1);
    Range<4> J(jfirst + 2, jlast - 1);
    Range<4> K(nk - 5, nk + 1);  // THIS WAS 6

    sycl::range<3> tpb(K.tpb, J.tpb, I.tpb);
    sycl::range<3> blocks(K.blocks, J.blocks, I.blocks);

    q.parallel_for(
        sycl::nd_range<3>(blocks * tpb, tpb), [=](sycl::nd_item<3> item) {
          kernel5(I.start, I.end, J.start, J.end, K.start, K.end, ifirst, ilast,
                  jfirst, jlast, kfirst, klast, nk, a1, sgn, d_u, d_mu,
                  d_lambda, d_met, d_jac, d_lu, d_acof, d_bope, d_ghcof,
                  d_acof_no_gp, d_ghcof_no_gp, d_strx, d_stry, item);
        });
  }
}
