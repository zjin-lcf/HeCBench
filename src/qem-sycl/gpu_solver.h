#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include <sycl/sycl.hpp>

void QRdel(sycl::queue &q, int n, float *A, float *B, float *C, float *D, float *b,
                      float *c, float *d, float *Q, float *R, float *Qint,
                      float *Rint, float *del);

void QuarticSolver(sycl::queue &q, int n, float *A, float *B, float *C, float *D,
                              float *b, float *Q, float *R, float *del,
                              float *theta, float *sqrtQ, float *x1, float *x2,
                              float *x3, float *temp, float *min);

void QuarticMinimumGPU(sycl::queue &q, int N, float *A, float *B, float *C, float *D, float *E,
                       float *min);

void QuarticMinimumGPUStreams(sycl::queue &q, int N, float *A, float *B, float *C, float *D,
                              float *E, float *min);

#endif
