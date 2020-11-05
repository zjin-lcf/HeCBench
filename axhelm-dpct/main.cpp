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
  o_ggeo = sycl::malloc_device<double>(Np * Nelements * p_Nggeo, q_ct1);
  o_q = sycl::malloc_device<double>(Ndim * Np * Nelements, q_ct1);
  o_Aq = sycl::malloc_device<double>(Ndim * Np * Nelements, q_ct1);
  o_DrV = sycl::malloc_device<double>(Nq * Nq, q_ct1);
  o_lambda = sycl::malloc_device<double>(2 * offset, q_ct1);

  q_ct1.memcpy(o_ggeo, ggeo, Np * Nelements * p_Nggeo * sizeof(dfloat)).wait();
  q_ct1.memcpy(o_q, q, (Ndim * Np) * Nelements * sizeof(dfloat)).wait();
  q_ct1.memcpy(o_DrV, DrV, Nq * Nq * sizeof(dfloat)).wait();
  q_ct1.memcpy(o_lambda, lambda, 2 * offset * sizeof(dfloat)).wait();

  for(int test=0;test<Ntests;++test) {
    if (Ndim > 1)
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

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, Nelements) *
                                  sycl::range<3>(1, 8, 8),
                              sycl::range<3>(1, 8, 8)),
            [=](sycl::nd_item<3> item_ct1) {
              axhelm_n3(
                  Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq,
                  item_ct1, s_D_acc_ct1.get_pointer(),
                  s_U_acc_ct1.get_pointer(), s_V_acc_ct1.get_pointer(),
                  s_W_acc_ct1.get_pointer(), s_GUr_acc_ct1.get_pointer(),
                  s_GUs_acc_ct1.get_pointer(), s_GVr_acc_ct1.get_pointer(),
                  s_GVs_acc_ct1.get_pointer(), s_GWr_acc_ct1.get_pointer(),
                  s_GWs_acc_ct1.get_pointer());
            });
      });
    else
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

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, Nelements) *
                                  sycl::range<3>(1, 8, 8),
                              sycl::range<3>(1, 8, 8)),
            [=](sycl::nd_item<3> item_ct1) {
              axhelm(Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq,
                     item_ct1, s_D_acc_ct1.get_pointer(),
                     s_q_acc_ct1.get_pointer(), s_Gqr_acc_ct1.get_pointer(),
                     s_Gqs_acc_ct1.get_pointer());
            });
      });
  }

  // store the device results in the 'q' array
  q_ct1.memcpy(q, o_Aq, Ndim * Np * Nelements * sizeof(dfloat)).wait();

  sycl::free(o_ggeo, q_ct1);
  sycl::free(o_q, q_ct1);
  sycl::free(o_Aq, q_ct1);
  sycl::free(o_DrV, q_ct1);
  sycl::free(o_lambda, q_ct1);

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
