bool verify (const double res,
             const int n,
             const double *__restrict__ x,
             const int nfun,
             const double *__restrict__ xfun)  
{
  double *linterp = (double*) malloc ( n * nfun * sizeof ( double ) );
  
  double lmax = 0.0;
  for (int j = 0; j < nfun; j++) {
    for (int i = 0; i < n; i++ )
      linterp[i*nfun+j] = 1.0;

    for (int i1 = 0; i1 < n; i1++ )
      for (int i2 = 0; i2 < n; i2++ )
        if ( i1 != i2 )
          linterp[i1*nfun+j] = linterp[i1*nfun+j] * ( xfun[j] - x[i2] ) / ( x[i1] - x[i2] );

    double t = 0.0;
    for (int i = 0; i < n; i++ )
      t += fabs ( linterp[i*nfun+j] );

    lmax = lmax > t ? lmax : t;
  }
  free(linterp);
  return fabs(res - lmax) <= 1e-6;
}
