#include <algorithm>
#include <stdio.h>
#include <sycl/sycl.hpp>

void QRdel(int n, const float *A, const float *B, const float *C,
           const float *D, float *__restrict__ b, float *__restrict__ c,
           float *__restrict__ d, float *__restrict__ Q, float *__restrict__ R,
           float *__restrict__ Qint, float *__restrict__ Rint,
           float *__restrict__ del, const sycl::nd_item<1> &item) {
  int i = item.get_global_id(0);
  if (i < n) {

    b[i] = 0.75f * (B[i] / A[i]);
    c[i] = 0.50f * (C[i] / A[i]);
    d[i] = 0.25f * (D[i] / A[i]);

    Q[i] = (c[i] / 3.f) - ((b[i] * b[i]) / 9.f);
    R[i] = (b[i] * c[i]) / 6.f - (b[i] * b[i] * b[i]) / 27.f - 0.5f * d[i];

    // round Q and R to get around problems caused by floating point precision
    Q[i] = sycl::round(Q[i] * 1E5f) / 1E5f;
    R[i] = sycl::round(R[i] * 1E5f) / 1E5f;

    Qint[i] = (Q[i] * Q[i] * Q[i]);
    Rint[i] = (R[i] * R[i]);

    del[i] = Rint[i] + Qint[i];
    // del[i] = (R[i] * R[i]) + (Q[i] * Q[i] * Q[i]); // why not just Q*Q*Q +
    // R*R? Heisenbug. Heisenbug in release code
  }
}

void QuarticSolver(int n, const float *A, const float *B, const float *C,
                   const float *D, const float *b, const float *Q,
                   const float *R, const float *del, float *__restrict__ theta,
                   float *__restrict__ sqrtQ, float *__restrict__ x1,
                   float *__restrict__ x2, float *__restrict__ x3,
                   float *__restrict__ temp, float *__restrict__ min,
                   const sycl::nd_item<1> &item) {
  // solver for finding minimum (xmin) for f(x) = Ax^4 + Bx^3 + Cx^2 + Dx + E
  // undefined behaviour if A=0

  int i = item.get_global_id(0);
  if (i < n) {
    // comparing against 1E-5 to deal with potential problems with comparing
    // floats to zero
    if (del[i] <= 1E-5f) { // all 3 roots real

      /*sqrtQ = 2 * sqrt(-Q);
         theta = acos(R / (sqrtQ ^ 3));

         x1 = 2 * (sqrtQ*cos(theta / 3) - b/3;
         x2 = 2 * (sqrtQ*cos((theta + 2 * pi) / 3) - b/3);
         x3 = 2 * (sqrtQ*cos((theta + 4 * pi) / 3) - b/3);*/

      theta[i] = sycl::acos((R[i] / sycl::sqrt(-(Q[i] * Q[i] * Q[i]))));
      sqrtQ[i] = 2.f * sycl::sqrt(-Q[i]);

      x1[i] = ((sqrtQ[i] * sycl::cos((theta[i]) / 3.f)) - (b[i] / 3.f));
      x2[i] = ((sqrtQ[i] * sycl::cos((theta[i] + 2.f * 3.1415927f) / 3.f)) -
               (b[i] / 3.f));
      x3[i] = ((sqrtQ[i] * sycl::cos((theta[i] + 4.f * 3.1415927f) / 3.f)) -
               (b[i] / 3.f));

      // unrolled bubble sort
      if (x1[i] < x2[i]) {
        temp[i] = x1[i];
        x1[i] = x2[i];
        x2[i] = temp[i];
      } // { swap(x1[i], x2[i]); }//swap
      if (x2[i] < x3[i]) {
        temp[i] = x2[i];
        x2[i] = x3[i];
        x3[i] = temp[i];
      } //{ swap(x2[i], x3[i]); }//swap
      if (x1[i] < x2[i]) {
        temp[i] = x1[i];
        x1[i] = x2[i];
        x2[i] = temp[i];
      } //{ swap(x1[i], x2[i]); }//swap

      min[i] =
          A[i] *
                          ((x1[i] * x1[i] * x1[i] * x1[i]) -
                           (x3[i] * x3[i] * x3[i] * x3[i])) /
                          4.f +
                      B[i] *
                          ((x1[i] * x1[i] * x1[i]) - (x3[i] * x3[i] * x3[i])) /
                          3.f +
                      C[i] * ((x1[i] * x1[i]) - (x3[i] * x3[i])) / 2.f +
                      D[i] * (x1[i] - x3[i]) <=
                  0.f
              ? x1[i]
              : x3[i];

    }

    // if (del[i] > 0) { // only 1 real root
    else {

      /*S = (R + sqrtD)^(1 / 3);
         T = (R - sqrtD)^(1 / 3);
         x = S + T - b/3;*/

      x1[i] = sycl::cbrt((R[i] + sycl::sqrt((float)(del[i])))) +
              sycl::cbrt((R[i] - sycl::sqrt((float)(del[i])))) -
              (b[i] / 3.f); // real root

      // complex conjugate roots not relevant for minimisation

      x2[i] = 0;
      x3[i] = 0;

      min[i] = x1[i];
    }
  }
}

void QuarticMinimumGPU(sycl::queue &q, int N, float *A, float *B, float *C,
                       float *D, float *E, float *min) {

  float *d_A, *d_B, *d_C, *d_D, *d_E, *d_bi, *d_ci, *d_di, *d_theta, *d_sqrtQ,
      *d_Q, *d_R, *d_Qint, *d_Rint, *d_del, *d_x1, *d_x2, *d_x3, *d_min,
      *d_temp;

  // kernel dims
  const int block_dim = 64;

  // device malloc
  d_A = sycl::malloc_device<float>(N, q);
  d_B = sycl::malloc_device<float>(N, q);
  d_C = sycl::malloc_device<float>(N, q);
  d_D = sycl::malloc_device<float>(N, q);
  d_E = sycl::malloc_device<float>(N, q);

  d_bi = sycl::malloc_device<float>(N, q);
  d_ci = sycl::malloc_device<float>(N, q);
  d_di = sycl::malloc_device<float>(N, q);
  d_theta = sycl::malloc_device<float>(N, q);
  d_sqrtQ = sycl::malloc_device<float>(N, q);

  d_Q = sycl::malloc_device<float>(N, q);
  d_R = sycl::malloc_device<float>(N, q);
  d_Qint = sycl::malloc_device<float>(N, q);
  d_Rint = sycl::malloc_device<float>(N, q);
  d_del = sycl::malloc_device<float>(N, q);

  d_x1 = sycl::malloc_device<float>(N, q);
  d_x2 = sycl::malloc_device<float>(N, q);
  d_x3 = sycl::malloc_device<float>(N, q);

  d_min = sycl::malloc_device<float>(N, q);

  d_temp = sycl::malloc_device<float>(N, q);

  q.memcpy(d_A, A, N * sizeof(float));
  q.memcpy(d_B, B, N * sizeof(float));
  q.memcpy(d_C, C, N * sizeof(float));
  q.memcpy(d_D, D, N * sizeof(float));

  sycl::range<1> gws((N + block_dim - 1) / block_dim * block_dim);
  sycl::range<1> lws(block_dim);

  q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
    QRdel(N, d_A, d_B, d_C, d_D, d_bi, d_ci, d_di, d_Q, d_R, d_Qint, d_Rint,
          d_del, item);
  });

  q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
    QuarticSolver(N, d_A, d_B, d_C, d_D, d_bi, d_Q, d_R, d_del, d_theta,
                  d_sqrtQ, d_x1, d_x2, d_x3, d_temp, d_min, item);
  });

  q.memcpy(min, d_min, N * sizeof(float)).wait();

  sycl::free(d_A, q);
  sycl::free(d_B, q);
  sycl::free(d_C, q);
  sycl::free(d_D, q);
  sycl::free(d_E, q);

  sycl::free(d_bi, q);
  sycl::free(d_ci, q);
  sycl::free(d_di, q);
  sycl::free(d_theta, q);
  sycl::free(d_sqrtQ, q);

  sycl::free(d_Q, q);
  sycl::free(d_R, q);
  sycl::free(d_Qint, q);
  sycl::free(d_Rint, q);
  sycl::free(d_del, q);

  sycl::free(d_x1, q);
  sycl::free(d_x2, q);
  sycl::free(d_x3, q);

  sycl::free(d_min, q);
  sycl::free(d_temp, q);
}

void QuarticMinimumGPUStreams(sycl::queue &q, int N, float *A, float *B,
                              float *C, float *D, float *E, float *min) {

  float *d_A, *d_B, *d_C, *d_D, *d_E, *d_bi, *d_ci, *d_di, *d_theta, *d_sqrtQ,
      *d_Q, *d_R, *d_Qint, *d_Rint, *d_del, *d_x1, *d_x2, *d_x3, *d_min,
      *d_temp;

  // kernel dims
  const int block_dim = 64;

  // initialize streams
  const int nStreams = 4;
  const int streamSize = N / nStreams;
  const int streamBytes = streamSize * sizeof(float);

  sycl::queue stream[nStreams + 1];

  for (int i = 0; i <= nStreams; ++i) {
#ifdef USE_GPU
    stream[i] =
        sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    stream[i] =
        sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  }

  int offset = 0;

  // device malloc
  d_A = sycl::malloc_device<float>(N, q);
  d_B = sycl::malloc_device<float>(N, q);
  d_C = sycl::malloc_device<float>(N, q);
  d_D = sycl::malloc_device<float>(N, q);
  d_E = sycl::malloc_device<float>(N, q);

  d_bi = sycl::malloc_device<float>(N, q);
  d_ci = sycl::malloc_device<float>(N, q);
  d_di = sycl::malloc_device<float>(N, q);
  d_theta = sycl::malloc_device<float>(N, q);
  d_sqrtQ = sycl::malloc_device<float>(N, q);

  d_Q = sycl::malloc_device<float>(N, q);
  d_R = sycl::malloc_device<float>(N, q);
  d_Qint = sycl::malloc_device<float>(N, q);
  d_Rint = sycl::malloc_device<float>(N, q);
  d_del = sycl::malloc_device<float>(N, q);

  d_x1 = sycl::malloc_device<float>(N, q);
  d_x2 = sycl::malloc_device<float>(N, q);
  d_x3 = sycl::malloc_device<float>(N, q);

  d_min = sycl::malloc_device<float>(N, q);

  d_temp = sycl::malloc_device<float>(N, q);

  for (int i = 0; i < nStreams; ++i) {
    offset = i * streamSize;
    stream[i].memcpy(&d_A[offset], &A[offset], streamBytes);
    stream[i].memcpy(&d_B[offset], &B[offset], streamBytes);
    stream[i].memcpy(&d_C[offset], &C[offset], streamBytes);
    stream[i].memcpy(&d_D[offset], &D[offset], streamBytes);

    float *d_A_offset = &d_A[offset];
    float *d_B_offset = &d_B[offset];
    float *d_C_offset = &d_C[offset];
    float *d_D_offset = &d_D[offset];
    float *d_bi_offset = &d_bi[offset];
    float *d_ci_offset = &d_ci[offset];
    float *d_di_offset = &d_di[offset];
    float *d_Q_offset = &d_Q[offset];
    float *d_R_offset = &d_R[offset];
    float *d_Qint_offset = &d_Qint[offset];
    float *d_Rint_offset = &d_Rint[offset];
    float *d_del_offset = &d_del[offset];
    float *d_theta_offset = &d_theta[offset];
    float *d_sqrtQ_offset = &d_sqrtQ[offset];
    float *d_x1_offset = &d_x1[offset];
    float *d_x2_offset = &d_x2[offset];
    float *d_x3_offset = &d_x3[offset];
    float *d_temp_offset = &d_temp[offset];
    float *d_min_offset = &d_min[offset];

    sycl::range<1> gws((streamSize + block_dim - 1) / block_dim * block_dim);
    sycl::range<1> lws(block_dim);

    stream[i].submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        QRdel(streamSize, d_A_offset, d_B_offset, d_C_offset, d_D_offset,
              d_bi_offset, d_ci_offset, d_di_offset, d_Q_offset, d_R_offset,
              d_Qint_offset, d_Rint_offset, d_del_offset, item);
      });
    });

    stream[i].submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        QuarticSolver(streamSize, d_A_offset, d_B_offset, d_C_offset,
                      d_D_offset, d_bi_offset, d_Q_offset, d_R_offset,
                      d_del_offset, d_theta_offset, d_sqrtQ_offset, d_x1_offset,
                      d_x2_offset, d_x3_offset, d_temp_offset, d_min_offset,
                      item);
      });
    });

    stream[i].memcpy(&min[offset], &d_min[offset], streamBytes);
  }

  const int resstreamSize = N % nStreams;
  const int resstreamBytes = resstreamSize * sizeof(float);
  if (resstreamSize != 0) {// Catch last bit of data from potential unequal
                           // division between streams
    offset = nStreams * streamSize;

    stream[nStreams].memcpy(&d_A[offset], &A[offset], resstreamBytes);
    stream[nStreams].memcpy(&d_B[offset], &B[offset], resstreamBytes);
    stream[nStreams].memcpy(&d_C[offset], &C[offset], resstreamBytes);
    stream[nStreams].memcpy(&d_D[offset], &D[offset], resstreamBytes);

    float *d_A_offset = &d_A[offset];
    float *d_B_offset = &d_B[offset];
    float *d_C_offset = &d_C[offset];
    float *d_D_offset = &d_D[offset];
    float *d_bi_offset = &d_bi[offset];
    float *d_ci_offset = &d_ci[offset];
    float *d_di_offset = &d_di[offset];
    float *d_Q_offset = &d_Q[offset];
    float *d_R_offset = &d_R[offset];
    float *d_Qint_offset = &d_Qint[offset];
    float *d_Rint_offset = &d_Rint[offset];
    float *d_del_offset = &d_del[offset];
    float *d_theta_offset = &d_theta[offset];
    float *d_sqrtQ_offset = &d_sqrtQ[offset];
    float *d_x1_offset = &d_x1[offset];
    float *d_x2_offset = &d_x2[offset];
    float *d_x3_offset = &d_x3[offset];
    float *d_temp_offset = &d_temp[offset];
    float *d_min_offset = &d_min[offset];

    sycl::range<1> gws((resstreamSize + block_dim - 1) / block_dim * block_dim);
    sycl::range<1> lws(block_dim);

    stream[nStreams].submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        QRdel(resstreamSize, d_A_offset, d_B_offset, d_C_offset, d_D_offset,
              d_bi_offset, d_ci_offset, d_di_offset, d_Q_offset, d_R_offset,
              d_Qint_offset, d_Rint_offset, d_del_offset, item);
      });
    });

    stream[nStreams].submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        QuarticSolver(resstreamSize, d_A_offset, d_B_offset, d_C_offset,
                      d_D_offset, d_bi_offset, d_Q_offset, d_R_offset,
                      d_del_offset, d_theta_offset, d_sqrtQ_offset, d_x1_offset,
                      d_x2_offset, d_x3_offset, d_temp_offset, d_min_offset,
                      item);
      });
    });

    stream[nStreams]
        .memcpy(&min[offset], &d_min[offset], resstreamBytes)
        .wait();
  }

  for (int i = 0; i < nStreams; i++) {
    stream[i].wait();
  }

  sycl::free(d_A, q);
  sycl::free(d_B, q);
  sycl::free(d_C, q);
  sycl::free(d_D, q);
  sycl::free(d_E, q);

  sycl::free(d_bi, q);
  sycl::free(d_ci, q);
  sycl::free(d_di, q);
  sycl::free(d_theta, q);
  sycl::free(d_sqrtQ, q);

  sycl::free(d_Q, q);
  sycl::free(d_R, q);
  sycl::free(d_Qint, q);
  sycl::free(d_Rint, q);
  sycl::free(d_del, q);

  sycl::free(d_x1, q);
  sycl::free(d_x2, q);
  sycl::free(d_x3, q);

  sycl::free(d_min, q);
  sycl::free(d_temp, q);
}
