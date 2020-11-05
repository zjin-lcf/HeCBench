#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define POLYNOMIAL_DEGREE  7
#define p_Nggeo 7
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_GWJID 0

#include "meshBasis.hpp"

// cpu reference 
#include "axhelmReference.cpp"

// gpu kernel
#include "axhelmKernel.cpp"
#include <cmath>

//=========================================================================
// End of kernel
//=========================================================================

dfloat *drandAlloc(int Nelem){

  dfloat *v = (dfloat*) calloc(Nelem, sizeof(dfloat));

  for(int n=0;n<Nelem;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  if (argc<4) {
    printf("Usage: ./axhelm Ndim numElements [nRepetitions]\n");
    return 1;
  }

  const int Ndim = atoi(argv[1]);
  const int Nelements = atoi(argv[2]);
  int Ntests = 1;
  if(argc>=4)
    Ntests = atoi(argv[3]);

  const int Nq = POLYNOMIAL_DEGREE + 1;
  const int Np = Nq*Nq*Nq;
  const int offset = Nelements*Np;


  // build element nodes and operators
  dfloat *rV, *wV, *DrV;
  meshJacobiGQ(0,0,POLYNOMIAL_DEGREE, &rV, &wV);
  meshDmatrix1D(POLYNOMIAL_DEGREE, Nq, rV, &DrV);

  std::cout << "word size: " << sizeof(dfloat) << " bytes\n";

  // populate device arrays
  dfloat *ggeo = drandAlloc(Np*Nelements*p_Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  const dfloat lambda1 = 1.1;
  dfloat *lambda = (dfloat*) calloc(2*offset, sizeof(dfloat));
  for(int i=0; i<offset; i++) {
    lambda[i]        = 1.0;
    lambda[i+offset] = lambda1;
  }

  // compute the reference result
  for(int n=0;n<Ndim;++n){
    dfloat *x = q + n*offset;
    dfloat *Ax = Aq + n*offset; 
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }

  auto start = std::chrono::high_resolution_clock::now();

  dfloat *o_ggeo, *o_q, *o_Aq, *o_DrV, *o_lambda;
  dpct::dpct_malloc((void **)&o_ggeo,
                    Np * Nelements * p_Nggeo * sizeof(dfloat));
  dpct::dpct_malloc((void **)&o_q, Ndim * Np * Nelements * sizeof(dfloat));
  dpct::dpct_malloc((void **)&o_Aq, Ndim * Np * Nelements * sizeof(dfloat));
  dpct::dpct_malloc((void **)&o_DrV, Nq * Nq * sizeof(dfloat));
  dpct::dpct_malloc((void **)&o_lambda, 2 * offset * sizeof(dfloat));

  dpct::dpct_memcpy(o_ggeo, ggeo, Np * Nelements * p_Nggeo * sizeof(dfloat),
                    dpct::host_to_device);
  dpct::dpct_memcpy(o_q, q, (Ndim * Np) * Nelements * sizeof(dfloat),
                    dpct::host_to_device);
  dpct::dpct_memcpy(o_DrV, DrV, Nq * Nq * sizeof(dfloat), dpct::host_to_device);
  dpct::dpct_memcpy(o_lambda, lambda, 2 * offset * sizeof(dfloat),
                    dpct::host_to_device);

  for(int test=0;test<Ntests;++test) {
    if (Ndim > 1)
    {
      dpct::buffer_t o_ggeo_buf_ct2 = dpct::get_buffer(o_ggeo);
      dpct::buffer_t o_DrV_buf_ct3 = dpct::get_buffer(o_DrV);
      dpct::buffer_t o_lambda_buf_ct4 = dpct::get_buffer(o_lambda);
      dpct::buffer_t o_q_buf_ct5 = dpct::get_buffer(o_q);
      dpct::buffer_t o_Aq_buf_ct6 = dpct::get_buffer(o_Aq);
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_D_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_U_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_V_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_W_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_GUr_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_GUs_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_GVr_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_GVs_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_GWr_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_GWs_acc_ct1(sycl::range<1>(64), cgh);
        auto o_ggeo_acc_ct2 =
            o_ggeo_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
        auto o_DrV_acc_ct3 =
            o_DrV_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
        auto o_lambda_acc_ct4 =
            o_lambda_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
        auto o_q_acc_ct5 =
            o_q_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
        auto o_Aq_acc_ct6 =
            o_Aq_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, Nelements) *
                                  sycl::range<3>(1, 8, 8),
                              sycl::range<3>(1, 8, 8)),
            [=](sycl::nd_item<3> item_ct1) {
              axhelm_n3(
                  Nelements, offset, (const double *)(&o_ggeo_acc_ct2[0]),
                  (const double *)(&o_DrV_acc_ct3[0]),
                  (const double *)(&o_lambda_acc_ct4[0]),
                  (const double *)(&o_q_acc_ct5[0]),
                  (double *)(&o_Aq_acc_ct6[0]), item_ct1,
                  s_D_acc_ct1.get_pointer(), s_U_acc_ct1.get_pointer(),
                  s_V_acc_ct1.get_pointer(), s_W_acc_ct1.get_pointer(),
                  s_GUr_acc_ct1.get_pointer(), s_GUs_acc_ct1.get_pointer(),
                  s_GVr_acc_ct1.get_pointer(), s_GVs_acc_ct1.get_pointer(),
                  s_GWr_acc_ct1.get_pointer(), s_GWs_acc_ct1.get_pointer());
            });
      });
    } else {
      dpct::buffer_t o_ggeo_buf_ct2 = dpct::get_buffer(o_ggeo);
      dpct::buffer_t o_DrV_buf_ct3 = dpct::get_buffer(o_DrV);
      dpct::buffer_t o_lambda_buf_ct4 = dpct::get_buffer(o_lambda);
      dpct::buffer_t o_q_buf_ct5 = dpct::get_buffer(o_q);
      dpct::buffer_t o_Aq_buf_ct6 = dpct::get_buffer(o_Aq);
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_D_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_q_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_Gqr_acc_ct1(sycl::range<1>(64), cgh);
        sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_Gqs_acc_ct1(sycl::range<1>(64), cgh);
        auto o_ggeo_acc_ct2 =
            o_ggeo_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
        auto o_DrV_acc_ct3 =
            o_DrV_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
        auto o_lambda_acc_ct4 =
            o_lambda_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
        auto o_q_acc_ct5 =
            o_q_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
        auto o_Aq_acc_ct6 =
            o_Aq_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, Nelements) *
                                  sycl::range<3>(1, 8, 8),
                              sycl::range<3>(1, 8, 8)),
            [=](sycl::nd_item<3> item_ct1) {
              axhelm(Nelements, offset, (const double *)(&o_ggeo_acc_ct2[0]),
                     (const double *)(&o_DrV_acc_ct3[0]),
                     (const double *)(&o_lambda_acc_ct4[0]),
                     (const double *)(&o_q_acc_ct5[0]),
                     (double *)(&o_Aq_acc_ct6[0]), item_ct1,
                     s_D_acc_ct1.get_pointer(), s_q_acc_ct1.get_pointer(),
                     s_Gqr_acc_ct1.get_pointer(), s_Gqs_acc_ct1.get_pointer());
            });
      });
    }
  }

  // store the device results in the 'q' array
  dpct::dpct_memcpy(q, o_Aq, Ndim * Np * Nelements * sizeof(dfloat),
                    dpct::device_to_host);

  dpct::dpct_free(o_ggeo);
  dpct::dpct_free(o_q);
  dpct::dpct_free(o_Aq);
  dpct::dpct_free(o_DrV);
  dpct::dpct_free(o_lambda);

  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

  // verify
  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = fabs(q[n] - Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  std::cout << "Correctness check: maxError = " << maxDiff << "\n";

  free(ggeo);
  free(q);
  free(Aq);
  free(lambda);
  free(rV);
  free(wV);
  free(DrV);

  // print statistics
  const dfloat GDOFPerSecond = Ndim*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*Nelements/elapsed;
  const long long bytesMoved = (Ndim*2*Np+7*Np+2*Np)*sizeof(dfloat); // x, Mx, opa, lambda
  const double bw = bytesMoved*Nelements/elapsed;
  double flopCount = Ndim*Np*12*Nq;
  if(Ndim == 1) flopCount += 22*Np;
  if(Ndim == 3) flopCount += 69*Np;
  double gflops = flopCount*Nelements/elapsed;
  std::cout << " NRepetitions=" << Ntests
    << " Ndim=" << Ndim
    << " N=" << POLYNOMIAL_DEGREE
    << " Nelements=" << Nelements
    << " elapsed time=" << elapsed
    << " GDOF/s=" << GDOFPerSecond
    << " GB/s=" << bw
    << " GFLOPS/s=" << gflops
    << "\n";
  exit(0);
}
