#include <sycl/sycl.hpp>

void ccsd_tengy_gpu(sycl::queue &q,
    const double * __restrict f1n,    const double * __restrict f1t,
    const double * __restrict f2n,    const double * __restrict f2t,
    const double * __restrict f3n,    const double * __restrict f3t,
    const double * __restrict f4n,    const double * __restrict f4t,
    const double * __restrict dintc1, const double * __restrict dintx1,const double * __restrict t1v1,
    const double * __restrict dintc2, const double * __restrict dintx2, const double * __restrict t1v2,
    const double * __restrict eorb,   const double eaijk,
    double * __restrict emp4i, double * __restrict emp5i,
    double * __restrict emp4k, double * __restrict emp5k,
    const int ncor, const int nocc, const int nvir);

void ccsd_trpdrv(sycl::queue &q,
    double * __restrict f1n, double * __restrict f1t,
    double * __restrict f2n, double * __restrict f2t,
    double * __restrict f3n, double * __restrict f3t,
    double * __restrict f4n, double * __restrict f4t,
    double * __restrict eorb,
    int    * __restrict ncor_, int * __restrict nocc_, int * __restrict nvir_,
    double * __restrict emp4_, double * __restrict emp5_,
    int    * __restrict a_, int * __restrict i_, int * __restrict j_, int * __restrict k_, int * __restrict klo_,
    double * __restrict tij, double * __restrict tkj, double * __restrict tia, double * __restrict tka,
    double * __restrict xia, double * __restrict xka, double * __restrict jia, double * __restrict jka,
    double * __restrict kia, double * __restrict kka, double * __restrict jij, double * __restrict jkj,
    double * __restrict kij, double * __restrict kkj,
    double * __restrict dintc1, double * __restrict dintx1, double * __restrict t1v1,
    double * __restrict dintc2, double * __restrict dintx2, double * __restrict t1v2)
{
  double emp4 = *emp4_;
  double emp5 = *emp5_;

  double emp4i = 0.0;
  double emp5i = 0.0;
  double emp4k = 0.0;
  double emp5k = 0.0;

  const int ncor = *ncor_;
  const int nocc = *nocc_;
  const int nvir = *nvir_;

  /* convert from Fortran to C offset convention... */
  const int k   = *k_ - 1;
  const int a   = *a_ - 1;
  const int i   = *i_ - 1;
  const int j   = *j_ - 1;

  const double eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

  ccsd_tengy_gpu(q, f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
      dintc1, dintx1, t1v1, dintc2, dintx2, t1v2,
      eorb, eaijk, &emp4i, &emp5i, &emp4k, &emp5k,
      ncor, nocc, nvir);

  emp4 += emp4i;
  emp5 += emp5i;

  if (*i_ != *k_) {
    emp4 += emp4k;
    emp5 += emp5k;
  }

  *emp4_ = emp4;
  *emp5_ = emp5;

  return;
}

