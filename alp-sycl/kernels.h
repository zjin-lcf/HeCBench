// LMAX is a fixed value for correctness
#define LMAX 4 
#define NDLM (LMAX*(LMAX+1)/2+LMAX)

//
// for the normalized associated Legendre functions \bar{P}_{lm}
// such that the spherical harmonics are:
// Y_lm(\theta, \phi) = \bar{P}_{lm}(\cos \theta) e^{i m \phi}
// use the recursion relation:
// P_{00}(x) = \sqrt{1/2\pi}
//
// i.e \bar{P}_{lm}=\sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}} P_{lm}
//
// Plm is a 1-d array that will contain all the values of P_{lm}(x) from P_{00} to P_{l_{max} l_{max}}
// the index into this array is Plm[l*(l+1)/2 + m]
//

inline int plmIdxDev(int l, int m)
{ return l*(l+1)/2+m; }

inline
void associatedLegendreFunctionNormalizedDevice(nd_item<1> &item, double x, int lmax, double *Plm)
{
  const double pi = sycl::acos(-1.0);
  double y = sycl::sqrt(1.0-x*x);
  int lid = item.get_local_id(0);
  int wgs = item.get_local_range(0);

  if (lid == 0) Plm[0]=sycl::sqrt(1.0/(4.0*pi));
  item.barrier(access::fence_space::local_space);

  if(lmax<1) return;

  for(int m=lid+1; m<=lmax; m+=wgs)
  {
    Plm[plmIdxDev(m,m)] = - sycl::sqrt(((double)(2*m+1)/(double)(2*m))) * y * Plm[plmIdxDev(m-1,m-1)];
    Plm[plmIdxDev(m,m-1)] = sycl::sqrt((double)(2*m+1)) * x * Plm[plmIdxDev(m-1,m-1)]; 
  }

  for(int m=lid; m<lmax; m+=wgs)
  {
    for(int l=m+2; l<=lmax; l++)
    {
      double a_lm = sycl::sqrt((double)(4*l*l-1)/(double)(l*l - m*m));
      double b_lm = sycl::sqrt((double)(((l-1)*(l-1) - m*m)/(double)(4*(l-1)*(l-1)-1)));
      Plm[plmIdxDev(l,m)] = a_lm * (x * Plm[plmIdxDev(l-1,m)] - b_lm * Plm[plmIdxDev(l-2,m)]);
    }
  }
}

void associatedLegendre(nd_item<1> &item, double x, int lmax, double *plm, double *plmOut)
{
  associatedLegendreFunctionNormalizedDevice(item,x,lmax,plm);
  for ( int i = item.get_local_id(0); i <= LMAX; i += item.get_local_range(0))
    plmOut[i] = plm[i];
}
