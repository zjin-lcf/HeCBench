#include <algorithm>
#include <stdio.h>
#include <hip/hip_runtime.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
hipError_t checkHip(hipError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
  }
#endif
  return result;
}

__global__ void QRdel(int n, const float *A, const float *B, const float *C,
                      const float *D, float *__restrict__ b,
                      float *__restrict__ c, float *__restrict__ d,
                      float *__restrict__ Q, float *__restrict__ R,
                      float *__restrict__ Qint, float *__restrict__ Rint,
                      float *__restrict__ del) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {

    b[i] = 0.75f * (B[i] / A[i]);
    c[i] = 0.50f * (C[i] / A[i]);
    d[i] = 0.25f * (D[i] / A[i]);

    Q[i] = (c[i] / 3.f) - ((b[i] * b[i]) / 9.f);
    R[i] = (b[i] * c[i]) / 6.f - (b[i] * b[i] * b[i]) / 27.f - 0.5f * d[i];

    // round Q and R to get around problems caused by floating point precision
    Q[i] = roundf(Q[i] * 1E5f) / 1E5f;
    R[i] = roundf(R[i] * 1E5f) / 1E5f;

    Qint[i] = (Q[i] * Q[i] * Q[i]);
    Rint[i] = (R[i] * R[i]);

    del[i] = Rint[i] + Qint[i];
    // del[i] = (R[i] * R[i]) + (Q[i] * Q[i] * Q[i]); // why not just Q*Q*Q +
    // R*R? Heisenbug. Heisenbug in release code
  }
}

__global__ void QuarticSolver(int n, const float *A, const float *B,
                              const float *C, const float *D, const float *b,
                              const float *Q, const float *R, const float *del,
                              float *__restrict__ theta,
                              float *__restrict__ sqrtQ, float *__restrict__ x1,
                              float *__restrict__ x2, float *__restrict__ x3,
                              float *__restrict__ temp,
                              float *__restrict__ min) {
  // solver for finding minimum (xmin) for f(x) = Ax^4 + Bx^3 + Cx^2 + Dx + E
  // undefined behaviour if A=0

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // comparing against 1E-5 to deal with potential problems with comparing
    // floats to zero
    if (del[i] <= 1E-5f) { // all 3 roots real

      /*sqrtQ = 2 * sqrt(-Q);
         theta = acos(R / (sqrtQ ^ 3));

         x1 = 2 * (sqrtQ*cos(theta / 3) - b/3;
         x2 = 2 * (sqrtQ*cos((theta + 2 * pi) / 3) - b/3);
         x3 = 2 * (sqrtQ*cos((theta + 4 * pi) / 3) - b/3);*/

      theta[i] = acosf((R[i] / sqrtf(-(Q[i] * Q[i] * Q[i]))));
      sqrtQ[i] = 2.f * sqrtf(-Q[i]);

      x1[i] = ((sqrtQ[i] * cosf((theta[i]) / 3.f)) - (b[i] / 3.f));
      x2[i] = ((sqrtQ[i] * cosf((theta[i] + 2.f * 3.1415927f) / 3.f)) -
               (b[i] / 3.f));
      x3[i] = ((sqrtQ[i] * cosf((theta[i] + 4.f * 3.1415927f) / 3.f)) -
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

      x1[i] = cbrtf((R[i] + sqrtf(del[i]))) + cbrtf((R[i] - sqrtf(del[i]))) -
              (b[i] / 3.f); // real root

      // complex conjugate roots not relevant for minimisation

      x2[i] = 0;
      x3[i] = 0;

      min[i] = x1[i];
    }
  }
}

void QuarticMinimumGPU(int N, float *A, float *B, float *C, float *D, float *E,
                       float *min) {

  float *d_A, *d_B, *d_C, *d_D, *d_E, *d_bi, *d_ci, *d_di, *d_theta, *d_sqrtQ,
      *d_Q, *d_R, *d_Qint, *d_Rint, *d_del, *d_x1, *d_x2, *d_x3, *d_min,
      *d_temp;

  // kernel dims
  const int block_dim = 64;

  // device malloc
  checkHip(hipMalloc(&d_A, N * sizeof(float)));
  checkHip(hipMalloc(&d_B, N * sizeof(float)));
  checkHip(hipMalloc(&d_C, N * sizeof(float)));
  checkHip(hipMalloc(&d_D, N * sizeof(float)));
  checkHip(hipMalloc(&d_E, N * sizeof(float)));

  checkHip(hipMalloc(&d_bi, N * sizeof(float)));
  checkHip(hipMalloc(&d_ci, N * sizeof(float)));
  checkHip(hipMalloc(&d_di, N * sizeof(float)));
  checkHip(hipMalloc(&d_theta, N * sizeof(float)));
  checkHip(hipMalloc(&d_sqrtQ, N * sizeof(float)));

  checkHip(hipMalloc(&d_Q, N * sizeof(float)));
  checkHip(hipMalloc(&d_R, N * sizeof(float)));
  checkHip(hipMalloc(&d_Qint, N * sizeof(float)));
  checkHip(hipMalloc(&d_Rint, N * sizeof(float)));
  checkHip(hipMalloc(&d_del, N * sizeof(float)));

  checkHip(hipMalloc(&d_x1, N * sizeof(float)));
  checkHip(hipMalloc(&d_x2, N * sizeof(float)));
  checkHip(hipMalloc(&d_x3, N * sizeof(float)));

  checkHip(hipMalloc(&d_min, N * sizeof(float)));

  checkHip(hipMalloc(&d_temp, N * sizeof(float)));

  hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_C, C, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_D, D, N * sizeof(float), hipMemcpyHostToDevice);

  QRdel<<<(N + block_dim - 1) / block_dim, block_dim>>>(
      N, d_A, d_B, d_C, d_D, d_bi, d_ci, d_di, d_Q, d_R, d_Qint, d_Rint, d_del);

  QuarticSolver<<<(N + block_dim - 1) / block_dim, block_dim>>>(
      N, d_A, d_B, d_C, d_D, d_bi, d_Q, d_R, d_del, d_theta, d_sqrtQ, d_x1,
      d_x2, d_x3, d_temp, d_min);

  hipMemcpy(min, d_min, N * sizeof(float), hipMemcpyDeviceToHost);

  checkHip(hipFree(d_A));
  checkHip(hipFree(d_B));
  checkHip(hipFree(d_C));
  checkHip(hipFree(d_D));
  checkHip(hipFree(d_E));

  checkHip(hipFree(d_bi));
  checkHip(hipFree(d_ci));
  checkHip(hipFree(d_di));
  checkHip(hipFree(d_theta));
  checkHip(hipFree(d_sqrtQ));

  checkHip(hipFree(d_Q));
  checkHip(hipFree(d_R));
  checkHip(hipFree(d_Qint));
  checkHip(hipFree(d_Rint));
  checkHip(hipFree(d_del));

  checkHip(hipFree(d_x1));
  checkHip(hipFree(d_x2));
  checkHip(hipFree(d_x3));

  checkHip(hipFree(d_min));
  checkHip(hipFree(d_temp));
}

void QuarticMinimumGPUStreams(int N, float *A, float *B, float *C, float *D,
                              float *E, float *min) {

  float *d_A, *d_B, *d_C, *d_D, *d_E, *d_bi, *d_ci, *d_di, *d_theta, *d_sqrtQ,
      *d_Q, *d_R, *d_Qint, *d_Rint, *d_del, *d_x1, *d_x2, *d_x3, *d_min,
      *d_temp;

  // kernel dims
  const int block_dim = 64;

  // initialize streams
  const int nStreams = 4;
  const int streamSize = N / nStreams;
  const int streamBytes = streamSize * sizeof(float);

  hipStream_t stream[nStreams + 1];

  for (int i = 0; i <= nStreams; ++i) {
    checkHip(hipStreamCreate(&stream[i]));
  }

  int offset = 0;

  // device malloc
  checkHip(hipMalloc(&d_A, N * sizeof(float)));
  checkHip(hipMalloc(&d_B, N * sizeof(float)));
  checkHip(hipMalloc(&d_C, N * sizeof(float)));
  checkHip(hipMalloc(&d_D, N * sizeof(float)));
  checkHip(hipMalloc(&d_E, N * sizeof(float)));

  checkHip(hipMalloc(&d_bi, N * sizeof(float)));
  checkHip(hipMalloc(&d_ci, N * sizeof(float)));
  checkHip(hipMalloc(&d_di, N * sizeof(float)));
  checkHip(hipMalloc(&d_theta, N * sizeof(float)));
  checkHip(hipMalloc(&d_sqrtQ, N * sizeof(float)));

  checkHip(hipMalloc(&d_Q, N * sizeof(float)));
  checkHip(hipMalloc(&d_R, N * sizeof(float)));
  checkHip(hipMalloc(&d_Qint, N * sizeof(float)));
  checkHip(hipMalloc(&d_Rint, N * sizeof(float)));
  checkHip(hipMalloc(&d_del, N * sizeof(float)));

  checkHip(hipMalloc(&d_x1, N * sizeof(float)));
  checkHip(hipMalloc(&d_x2, N * sizeof(float)));
  checkHip(hipMalloc(&d_x3, N * sizeof(float)));

  checkHip(hipMalloc(&d_min, N * sizeof(float)));

  checkHip(hipMalloc(&d_temp, N * sizeof(float)));

  for (int i = 0; i < nStreams; ++i) {
    offset = i * streamSize;
    hipMemcpyAsync(&d_A[offset], &A[offset], streamBytes,
                    hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync(&d_B[offset], &B[offset], streamBytes,
                    hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync(&d_C[offset], &C[offset], streamBytes,
                    hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync(&d_D[offset], &D[offset], streamBytes,
                    hipMemcpyHostToDevice, stream[i]);

    QRdel<<<(streamSize + block_dim - 1) / block_dim, block_dim, 0,
            stream[i]>>>(streamSize, &d_A[offset], &d_B[offset], &d_C[offset],
                         &d_D[offset], &d_bi[offset], &d_ci[offset],
                         &d_di[offset], &d_Q[offset], &d_R[offset],
                         &d_Qint[offset], &d_Rint[offset], &d_del[offset]);

    QuarticSolver<<<(streamSize + block_dim - 1) / block_dim, block_dim, 0,
                    stream[i]>>>(
        streamSize, &d_A[offset], &d_B[offset], &d_C[offset], &d_D[offset],
        &d_bi[offset], &d_Q[offset], &d_R[offset], &d_del[offset],
        &d_theta[offset], &d_sqrtQ[offset], &d_x1[offset], &d_x2[offset],
        &d_x3[offset], &d_temp[offset], &d_min[offset]);

    hipMemcpyAsync(&min[offset], &d_min[offset], streamBytes,
                    hipMemcpyDeviceToHost, stream[i]);
  }

  const int resstreamSize = N % nStreams;
  const int resstreamBytes = resstreamSize * sizeof(float);
  if (resstreamSize != 0) {// Catch last bit of data from potential unequal
                           // division between streams
    offset = nStreams * streamSize;

    hipMemcpyAsync(&d_A[offset], &A[offset], resstreamBytes,
                    hipMemcpyHostToDevice, stream[nStreams]);
    hipMemcpyAsync(&d_B[offset], &B[offset], resstreamBytes,
                    hipMemcpyHostToDevice, stream[nStreams]);
    hipMemcpyAsync(&d_C[offset], &C[offset], resstreamBytes,
                    hipMemcpyHostToDevice, stream[nStreams]);
    hipMemcpyAsync(&d_D[offset], &D[offset], resstreamBytes,
                    hipMemcpyHostToDevice, stream[nStreams]);

    QRdel<<<(resstreamSize + block_dim - 1) / block_dim, block_dim, 0,
            stream[nStreams]>>>(
        resstreamSize, &d_A[offset], &d_B[offset], &d_C[offset], &d_D[offset],
        &d_bi[offset], &d_ci[offset], &d_di[offset], &d_Q[offset], &d_R[offset],
        &d_Qint[offset], &d_Rint[offset], &d_del[offset]);

    QuarticSolver<<<(resstreamSize + block_dim - 1) / block_dim, block_dim, 0,
                    stream[nStreams]>>>(
        resstreamSize, &d_A[offset], &d_B[offset], &d_C[offset], &d_D[offset],
        &d_bi[offset], &d_Q[offset], &d_R[offset], &d_del[offset],
        &d_theta[offset], &d_sqrtQ[offset], &d_x1[offset], &d_x2[offset],
        &d_x3[offset], &d_temp[offset], &d_min[offset]);

    hipMemcpyAsync(&min[offset], &d_min[offset], resstreamBytes,
                    hipMemcpyDeviceToHost, stream[nStreams]);
  }

  hipDeviceSynchronize();

  checkHip(hipFree(d_A));
  checkHip(hipFree(d_B));
  checkHip(hipFree(d_C));
  checkHip(hipFree(d_D));
  checkHip(hipFree(d_E));

  checkHip(hipFree(d_bi));
  checkHip(hipFree(d_ci));
  checkHip(hipFree(d_di));
  checkHip(hipFree(d_theta));
  checkHip(hipFree(d_sqrtQ));

  checkHip(hipFree(d_Q));
  checkHip(hipFree(d_R));
  checkHip(hipFree(d_Qint));
  checkHip(hipFree(d_Rint));
  checkHip(hipFree(d_del));

  checkHip(hipFree(d_x1));
  checkHip(hipFree(d_x2));
  checkHip(hipFree(d_x3));

  checkHip(hipFree(d_min));
  checkHip(hipFree(d_temp));

  for (int i = 0; i <= nStreams; ++i) {
    checkHip(hipStreamDestroy(stream[i]));
  }
}
