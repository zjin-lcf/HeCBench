#define one_over_theta0  (5.0/(2.0*pow(M_PI,4.0)))
#define kappa (pow(M_PI,2.0)/3.0 - 2 * 1.2020569031595942)

#pragma omp declare target
/*
 there are some limitation on L_max
 due to GSL's factorial N! is only defined in GSL for N<=170
 thus L_max < 85 
 */
#define L_max 64

double B(int l)
{
  return 2.0*(14.0*l*l + 7.0*l - 2.0)/(4.0*l-1.0)/(4.0*l+3.0);
}

double U(int l)
{
  return -(2.0*l-1)*(2.0*l+1)*(2.0*l+2)/(4.0*l+3)/(4.0*l+5);
}

double C(int l)
{
  return (2.0*l-1)*2.0*l*(2.0*l+2)/(4.0*l-3.0)/(4.0*l-1.0);
}

double alpha(int l, const double* dftab, const double* ftab)
{
  if(l<0) return 0.0;
  if(l==0) return 1.0;
  else 
    return double_fact_table[2*l-1] / fact_table[l];
}

double Omega(int l, int m, int n, const double* dftab, const double* ftab)
{
  return alpha(m-n+l, dftab, ftab) * alpha(m+n-l, dftab, ftab) *
         alpha(n-m+l, dftab, ftab) / alpha(m+n+l, dftab, ftab) * 
         (4*l+1) / (2.0*(n+m+l)+1) ;
}

double Sum_Omega(int l, const double c[], const double* dftab, const double* ftab)
{
  double sum = 0.0;
  for(int m=1; m<L_max; m++)
    for(int n=1; n<L_max; n++)
      if(abs(m-n)<l+1) sum += Omega(l,m,n,dftab,ftab) * c[m]*c[n];
  return sum;
}

double Sum_NL(int l, const double c[])
{
  double sum = 0.0;
  for(int n=1; n<L_max; n++)
    sum += pow(c[n],2.0)/(4.0*n+1);
  return sum * (2*l-1)*(l+1)*c[l]/3.0;
}
#pragma omp end declare target

void RHS_f (const double* dftab, const double* ftab,
            double t, const double *c, double *RHS)
{
  #pragma omp target teams distribute parallel for num_teams(1) thread_limit(96)
  for (int l = 0; l < L_max; l++) {

    double T = c[0];

    if (l == 0) { 
      RHS[0] = - T/3.0/t *(1.0+0.1*c[1]);
    } else {
      double B_bar = B(l) - 4.0/3.0;
      double LHS_119;
      if (l > 1) 
        LHS_119 = 1.0/t * ( U(l) * c[l+1] + (B_bar - 2.0/15.0 * c[1]) + C(l) * c[l-1]);
      else 
        LHS_119 = 1.0/t * ( U(1) * c[2] + (B_bar - 2.0/15.0 * c[1]) + C(1));

      double Sum1 = Sum_Omega(l, c, dftab, ftab);
      double Sum2 = Sum_NL(l, c);

      double RHS_119 = - T*one_over_theta0 *(
          (kappa + M_PI*M_PI*l*(2*l+1)/3.0)*c[l] + 
           kappa*Sum1 + kappa*Sum2);

      RHS[l] = -LHS_119 + RHS_119;
    }
  }
}
