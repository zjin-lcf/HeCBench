void kernel_ref (const double * __restrict__ f1n,    const double * __restrict__ f1t,
                 const double * __restrict__ f2n,    const double * __restrict__ f2t,
                 const double * __restrict__ f3n,    const double * __restrict__ f3t,
                 const double * __restrict__ f4n,    const double * __restrict__ f4t,
                 const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                 const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                 const double * __restrict__ eorb,   const double eaijk,
                 double &emp4i, double &emp5i,
                 double &emp4k, double &emp5k,
                 const int ncor, const int nocc, const int nvir)
{
  for (int c = 0; c < nvir; c++) {
    for (int b = 0; b < nvir; b++) {
      const double denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

      // nvir < 10000 so this should never overflow
      const int bc = b+c*nvir;
      const int cb = c+b*nvir;

      const double f1nbc = f1n[bc];
      const double f1tbc = f1t[bc];
      const double f1ncb = f1n[cb];
      const double f1tcb = f1t[cb];

      const double f2nbc = f2n[bc];
      const double f2tbc = f2t[bc];
      const double f2ncb = f2n[cb];
      const double f2tcb = f2t[cb];

      const double f3nbc = f3n[bc];
      const double f3tbc = f3t[bc];
      const double f3ncb = f3n[cb];
      const double f3tcb = f3t[cb];

      const double f4nbc = f4n[bc];
      const double f4tbc = f4t[bc];
      const double f4ncb = f4n[cb];
      const double f4tcb = f4t[cb];

      emp4i += denom * (f1tbc+f1ncb+f2tcb+f3nbc+f4ncb) * (f1tbc-f2tbc*2-f3tbc*2+f4tbc)
                        - denom * (f1nbc+f1tcb+f2ncb+f3ncb) * (f1tbc*2-f2tbc-f3tbc+f4tbc*2)
                        + denom * 3 * (f1nbc*(f1nbc+f3ncb+f4tcb*2) +f2nbc*f2tcb+f3nbc*f4tbc);

      emp4k += denom * (f1nbc+f1tcb+f2ncb+f3tbc+f4tcb) * (f1nbc-f2nbc*2-f3nbc*2+f4nbc)
                        - denom * (f1tbc+f1ncb+f2tcb+f3tcb) * (f1nbc*2-f2nbc-f3nbc+f4nbc*2)
                        + denom * 3 * (f1tbc*(f1tbc+f3tcb+f4ncb*2) +f2tbc*f2ncb+f3tbc*f4nbc);

      const double t1v1b = t1v1[b];
      const double t1v2b = t1v2[b];

      const double dintx1c = dintx1[c];
      const double dintx2c = dintx2[c];
      const double dintc1c = dintc1[c];
      const double dintc2c = dintc2[c];

      emp5i += denom * t1v1b * dintx1c * (f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
                       +(f3nbc+f4tbc+f1ncb)*4)
                       + denom * t1v1b * dintc1c * (f1nbc+f4nbc+f1tcb -(f2nbc+f3nbc+f2tcb)*2);
      emp5k += denom * t1v2b * dintx2c * (f1nbc+f2tbc+f4tcb -(f3nbc+f4tbc+f2tcb +f1tbc+f2nbc+f3tcb)*2
                       +(f3tbc+f4nbc+f1tcb)*4)
                       + denom * t1v2b * dintc2c * (f1tbc+f4tbc+f1ncb -(f2tbc+f3tbc+f2ncb)*2);
    }
  }
}

void ccsd_tengy_ref(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                    const double * __restrict__ f2n,    const double * __restrict__ f2t,
                    const double * __restrict__ f3n,    const double * __restrict__ f3t,
                    const double * __restrict__ f4n,    const double * __restrict__ f4t,
                    const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                    const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                    const double * __restrict__ eorb,   const double eaijk,
                    double &emp4i_, double &emp5i_,
                    double &emp4k_, double &emp5k_,
                    const int ncor, const int nocc, const int nvir)
{
  double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

  kernel_ref(
        f1n,
        f1t,
        f2n,
        f2t,
        f3n,
        f3t,
        f4n,
        f4t,
        dintc1,
        dintx1,
        t1v1,
        dintc2,
        dintx2,
        t1v2,
        eorb, 
        eaijk,
        emp4i, 
        emp5i, 
        emp4k,
        emp5k, 
        ncor, nocc, nvir);

  emp4i_ = emp4i;
  emp4k_ = emp4k;
  emp5i_ = emp5i;
  emp5k_ = emp5k;
}

void ccsd_trpdrv_ref(double * __restrict__ f1n, double * __restrict__ f1t,
                     double * __restrict__ f2n, double * __restrict__ f2t,
                     double * __restrict__ f3n, double * __restrict__ f3t,
                     double * __restrict__ f4n, double * __restrict__ f4t,
                     double * __restrict__ eorb,
                     int    * __restrict__ ncor_, int * __restrict__ nocc_, int * __restrict__ nvir_,
                     double * __restrict__ emp4_, double * __restrict__ emp5_,
                     int    * __restrict__ a_, int * __restrict__ i_, int * __restrict__ j_, int * __restrict__ k_, int * __restrict__ klo_,
                     double * __restrict__ tij, double * __restrict__ tkj, double * __restrict__ tia, double * __restrict__ tka,
                     double * __restrict__ xia, double * __restrict__ xka, double * __restrict__ jia, double * __restrict__ jka,
                     double * __restrict__ kia, double * __restrict__ kka, double * __restrict__ jij, double * __restrict__ jkj,
                     double * __restrict__ kij, double * __restrict__ kkj,
                     double * __restrict__ dintc1, double * __restrict__ dintx1, double * __restrict__ t1v1,
                     double * __restrict__ dintc2, double * __restrict__ dintx2, double * __restrict__ t1v2)
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
    //const int klo = *klo_ - 1;
    const int a   = *a_ - 1;
    const int i   = *i_ - 1;
    const int j   = *j_ - 1;

    const double eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

    ccsd_tengy_ref(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
                   dintc1, dintx1, t1v1, dintc2, dintx2, t1v2,
                   eorb, eaijk, emp4i, emp5i, emp4k, emp5k,
                   ncor, nocc, nvir);

    emp4 += emp4i;
    emp5 += emp5i;

    if (*i_ != *k_) {
        emp4 += emp4k;
        emp5 += emp5k;
    }

    *emp4_ = emp4;
    *emp5_ = emp5;
}
