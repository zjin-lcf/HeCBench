//  SW4 LICENSE
// # ----------------------------------------------------------------------
// # SW4 - Seismic Waves, 4th order
// # ----------------------------------------------------------------------
// # Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
// # Produced at the Lawrence Livermore National Laboratory. 
// # 
// # Written by:
// # N. Anders Petersson (petersson1@llnl.gov)
// # Bjorn Sjogreen      (sjogreen2@llnl.gov)
// # 
// # LLNL-CODE-643337 
// # 
// # All rights reserved. 
// # 
// # This file is part of SW4, Version: 1.0
// # 
// # Please also read LICENCE.txt, which contains "Our Notice and GNU General Public License"
// # 
// # This program is free software; you can redistribute it and/or modify
// # it under the terms of the GNU General Public License (as published by
// # the Free Software Foundation) version 2, dated June 1991. 
// # 
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// # conditions of the GNU General Public License for more details. 
// # 
// # You should have received a copy of the GNU General Public License
// # along with this program; if not, write to the Free Software
// # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "mpi.h"
#include "GridPointSource.h"
#include "Require.h"

#include <fenv.h>
#include <cmath>

//#include  "EW.h"

using namespace std;

#include "time_functions_cu.h"
//-----------------------------------------------------------------------
GridPointSource::GridPointSource( float_sw4 frequency, float_sw4 t0,
				  int N, int M, int L, int G,
				  float_sw4 Fx, float_sw4 Fy, float_sw4 Fz,
				  timeDep tDep,
				  int ncyc, float_sw4* pars, int npar, int* ipars, int nipar,
				  float_sw4* devpars, int* devipars,
				  float_sw4* jacobian,
				  float_sw4* dddp, float_sw4* hess1, float_sw4* hess2, float_sw4* hess3 ):
  mFreq(frequency),
  mT0(t0),
  m_i0(N), m_j0(M), m_k0(L),m_grid(G),
  mTimeDependence(tDep),
  mNcyc(ncyc)
{
   // Copy only pointers, not memory.
   // mPar, mIpar will not be changed by this class, they should
   // be set correctly in Source.

  mPar   = pars;
  mNpar  = npar;
  mIpar  = ipars;
  mNipar = nipar;
  mdevPar  = devpars;
  mdevIpar = devipars;

  mForces[0] = Fx;
  mForces[1] = Fy;
  mForces[2] = Fz;

  //  cout << "Size of time function pointer " << sizeof(C6SmoothBump) << endl;
  m_derivative = -1;

  m_jacobian_known = jacobian != NULL;
  if( jacobian != NULL )
     for( int m=0 ; m < 27 ; m++ )
	m_jacobian[m] = jacobian[m];

  m_hessian_known = false;
  if( dddp != NULL && hess1 != NULL && hess2 != NULL && hess3 != NULL )
  {
     for( int m=0 ; m < 9 ; m++ )
     {
	m_dddp[m] = dddp[m];
        m_hesspos1[m] = hess1[m];
        m_hesspos2[m] = hess2[m];
        m_hesspos3[m] = hess3[m];
     }
     m_hessian_known = true;
  }
}

//-----------------------------------------------------------------------
GridPointSource::GridPointSource()
{ }

//-----------------------------------------------------------------------
GridPointSource::~GridPointSource()
{
  //   if( mPar != NULL )
   //  delete[] mPar;
}


//-----------------------------------------------------------------------
SYCL_EXTERNAL
void GridPointSource::getFxyz( float_sw4 t, float_sw4* fxyz ) const
{
   float_sw4 afun;
   afun= C6SmoothBump(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar );
      
  if( m_derivative==-1)
  {
	fxyz[0] = mForces[0]*afun;
	fxyz[1] = mForces[1]*afun;
	fxyz[2] = mForces[2]*afun;
  }
  else if( m_derivative >= 0 && m_derivative <= 8 )
  {
     fxyz[0] = m_jacobian[m_derivative*3]*afun;
     fxyz[1] = m_jacobian[m_derivative*3+1]*afun;
     fxyz[2] = m_jacobian[m_derivative*3+2]*afun;
  }
  else if( m_derivative == 9 )
  {
     afun = -C6SmoothBump_t(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] = mForces[0]*afun;
     fxyz[1] = mForces[1]*afun;
     fxyz[2] = mForces[2]*afun;
  }
  else if( m_derivative == 10 )
  {
     afun = C6SmoothBump_om(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] = mForces[0]*afun;
     fxyz[1] = mForces[1]*afun;
     fxyz[2] = mForces[2]*afun;
  }
  else if( m_derivative == 11 )
  {
     fxyz[0]= fxyz[1]=fxyz[2]=0;
     int i;
     for( i=0 ; i < 9; i++ )
     {
        fxyz[0] += afun*m_jacobian[i*3]*m_dir[i];
        fxyz[1] += afun*m_jacobian[i*3+1]*m_dir[i];
        fxyz[2] += afun*m_jacobian[i*3+2]*m_dir[i];
     }
     i = 9;
     afun = -C6SmoothBump_t(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] += afun*mForces[0]*m_dir[i];
     fxyz[1] += afun*mForces[1]*m_dir[i];
     fxyz[2] += afun*mForces[2]*m_dir[i];
     i = 10;
     afun =  C6SmoothBump_om(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] += afun*mForces[0]*m_dir[i];
     fxyz[1] += afun*mForces[1]*m_dir[i];
     fxyz[2] += afun*mForces[2]*m_dir[i];
  }
}

//-----------------------------------------------------------------------
void GridPointSource::getFxyz_notime( float_sw4* fxyz ) const
{
// For source spatial discretization testing
  fxyz[0] = mForces[0];
  fxyz[1] = mForces[1];
  fxyz[2] = mForces[2];
}

//-----------------------------------------------------------------------
SYCL_EXTERNAL
void GridPointSource::getFxyztt( float_sw4 t, float_sw4* fxyz ) const
{
   float_sw4 afun;
   afun= C6SmoothBump_tt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar );

  if( m_derivative==-1)
  {
	fxyz[0] = mForces[0]*afun;
	fxyz[1] = mForces[1]*afun;
	fxyz[2] = mForces[2]*afun;
  }
  else if( m_derivative >= 0 && m_derivative <= 8 )
  {
     fxyz[0] = m_jacobian[m_derivative*3]*afun;
     fxyz[1] = m_jacobian[m_derivative*3+1]*afun;
     fxyz[2] = m_jacobian[m_derivative*3+2]*afun;
  }
  else if( m_derivative == 9 )
  {
     afun = -C6SmoothBump_ttt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] = mForces[0]*afun;
     fxyz[1] = mForces[1]*afun;
     fxyz[2] = mForces[2]*afun;
  }
  else if( m_derivative == 10 )
  {
     afun = C6SmoothBump_omtt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] = mForces[0]*afun;
     fxyz[1] = mForces[1]*afun;
     fxyz[2] = mForces[2]*afun;
  }
  else if( m_derivative == 11 )
  {
     fxyz[0]= fxyz[1]=fxyz[2]=0;
     int i;
     for( i=0 ; i < 9; i++ )
     {
        fxyz[0] += afun*m_jacobian[i*3]*m_dir[i];
        fxyz[1] += afun*m_jacobian[i*3+1]*m_dir[i];
        fxyz[2] += afun*m_jacobian[i*3+2]*m_dir[i];
     }
     i = 9;
     afun = -C6SmoothBump_ttt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] += afun*mForces[0]*m_dir[i];
     fxyz[1] += afun*mForces[1]*m_dir[i];
     fxyz[2] += afun*mForces[2]*m_dir[i];
     i = 10;
     afun =  C6SmoothBump_omtt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
     fxyz[0] += afun*mForces[0]*m_dir[i];
     fxyz[1] += afun*mForces[1]*m_dir[i];
     fxyz[2] += afun*mForces[2]*m_dir[i];
  }
}


//-----------------------------------------------------------------------
void GridPointSource::set_derivative( int der, const float_sw4 dir[11] )
{
   if( der >= 0 && der <= 11 )
      m_derivative = der;
   for( int i=0 ; i < 11 ; i++ )
      m_dir[i] = dir[i];
}

//-----------------------------------------------------------------------
void GridPointSource::set_noderivative( )
{
   m_derivative = -1;
}

//-----------------------------------------------------------------------
void GridPointSource::limitFrequency(float_sw4 max_freq)
{
  if (mFreq > max_freq)
    mFreq=max_freq;
}

//-----------------------------------------------------------------------
float_sw4 GridPointSource::getTimeFunc(float_sw4 t) const
{
  return C6SmoothBump(mFreq, t - mT0, mPar, mNpar, mIpar, mNipar);
}

//-----------------------------------------------------------------------
float_sw4 GridPointSource::evalTimeFunc_t(float_sw4 t) const
{
  return C6SmoothBump_t(mFreq, t - mT0, mPar, mNpar, mIpar, mNipar);
}

//-----------------------------------------------------------------------
float_sw4 GridPointSource::evalTimeFunc_tt(float_sw4 t) const
{
  return C6SmoothBump_tt(mFreq, t - mT0, mPar, mNpar, mIpar, mNipar);
}

//-----------------------------------------------------------------------
float_sw4 GridPointSource::evalTimeFunc_ttt(float_sw4 t) const
{
  return C6SmoothBump_ttt(mFreq, t - mT0, mPar, mNpar, mIpar, mNipar);
}

//-----------------------------------------------------------------------
float_sw4 GridPointSource::evalTimeFunc_tttt(float_sw4 t) const
{
  return Gaussian_tttt(mFreq, t - mT0, mPar, mNpar, mIpar, mNipar);
}

//-----------------------------------------------------------------------
ostream& operator<<( ostream& output, const GridPointSource& s )
{
   output << "GridPointSource at (i,j,k) = " << s.m_i0 << "," << s.m_j0 << "," << s.m_k0 << 
     " in grid no " << s.m_grid << endl;
   //   output << "   Strength " << s.mAmp;
   output << " Fx Fy Fz = " << s.mForces[0] << " " << s.mForces[1] << " " << s.mForces[2] << endl;
   output << " freq = " << s.mFreq << " t0 = " <<  s.mT0 << endl;
   output << " npar = " <<  s.mNpar << " nipar= " <<  s.mNipar  << endl;
   if(  s.mNpar > 0 )
      output << " mpar[0] = " <<  s.mPar[0];
   if(  s.mNipar > 0 )
      output << " mipar[0] = " <<  s.mIpar[0];
   if(  s.mNpar >0 ||  s.mNipar > 0 )
      cout << endl;
   

   return output;
}

//-----------------------------------------------------------------------
void GridPointSource::add_to_gradient( std::vector<Sarray> & kappa, std::vector<Sarray> & eta,
				       float_sw4 t, float_sw4 dt, float_sw4 gradient[11], std::vector<float_sw4> & h,
				       Sarray& Jac, bool topography_exists )
{
   if( m_jacobian_known )
   {
      float_sw4 normwgh[4]={17.0/48.0, 59.0/48.0, 43.0/48.0, 49.0/48.0 };
      float_sw4 dt2o12 = dt*dt/12.0;
      float_sw4 g0= C6SmoothBump( mFreq, t-mT0, mPar, mNpar, mIpar, mNipar );
      float_sw4 g = g0 + dt2o12*C6SmoothBump_tt( mFreq, t-mT0, mPar, mNpar, mIpar, mNipar);

      // save some work by accessing array elements only once:
      float_sw4 kap1 = kappa[m_grid](1,m_i0,m_j0,m_k0);
      float_sw4 kap2 = kappa[m_grid](2,m_i0,m_j0,m_k0);
      float_sw4 kap3 = kappa[m_grid](3,m_i0,m_j0,m_k0);
      float_sw4 eta1 = eta[m_grid](1,m_i0,m_j0,m_k0);
      float_sw4 eta2 = eta[m_grid](2,m_i0,m_j0,m_k0);
      float_sw4 eta3 = eta[m_grid](3,m_i0,m_j0,m_k0);
      float_sw4 h3   = h[m_grid]*h[m_grid]*h[m_grid];
      if( topography_exists && m_grid == h.size()-1 )
	 h3 = Jac(m_i0,m_j0,m_k0);
      //      float_sw4 h3 = 1.0;
      if( 1 <= m_k0 && m_k0 <= 4 )
	 h3 *= normwgh[m_k0-1];

      // derivative wrt. position (m=0,1,2) and moment tensor components (m=3,..,8)
      for( int m= 0 ; m < 9 ; m++ )
      {
	 gradient[m] -= g*(  kap1*m_jacobian[3*m] + kap2*m_jacobian[3*m+1] +
				  kap3*m_jacobian[3*m+2]  )*h3;
	 gradient[m] -= dt2o12*g0*( eta1*m_jacobian[3*m] + eta2*m_jacobian[3*m+1] +
					 eta3*m_jacobian[3*m+2]  )*h3;
      }

      // derivative wrt. (t0, freq)
      float_sw4 dgt0 = -C6SmoothBump_t(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      float_sw4 dgom =  C6SmoothBump_om(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      gradient[9]   -= dt2o12*dgt0*( eta1*mForces[0] + eta2*mForces[1] + eta3*mForces[2])*h3;
      gradient[10]  -= dt2o12*dgom*( eta1*mForces[0] + eta2*mForces[1] + eta3*mForces[2])*h3;

      dgt0 = dgt0 - dt2o12*C6SmoothBump_ttt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      dgom = dgom + dt2o12*C6SmoothBump_omtt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      gradient[9]  -= dgt0*( kap1*mForces[0] + kap2*mForces[1] + kap3*mForces[2])*h3;
      gradient[10] -= dgom*( kap1*mForces[0] + kap2*mForces[1] + kap3*mForces[2])*h3;
   }
}

//-----------------------------------------------------------------------
void GridPointSource::add_to_hessian( std::vector<Sarray> & kappa, std::vector<Sarray> & eta,
				      float_sw4 t, float_sw4 dt, float_sw4 hessian[121],
				      std::vector<float_sw4> & h )
// Add upper part of symmetric matrix
{
   if( m_hessian_known && m_jacobian_known )
   {
      float_sw4 normwgh[4]={17.0/48.0, 59.0/48.0, 43.0/48.0, 49.0/48.0 };
      float_sw4 dt2o12 = dt*dt/12.0;
      float_sw4 g0= C6SmoothBump( mFreq, t-mT0, mPar, mNpar, mIpar, mNipar );
      float_sw4 g = g0 + dt2o12*C6SmoothBump_tt( mFreq, t-mT0, mPar, mNpar, mIpar, mNipar);

      // save some work by accessing array elements only once:
      float_sw4 kap1 = kappa[m_grid](1,m_i0,m_j0,m_k0);
      float_sw4 kap2 = kappa[m_grid](2,m_i0,m_j0,m_k0);
      float_sw4 kap3 = kappa[m_grid](3,m_i0,m_j0,m_k0);
      float_sw4 eta1 = eta[m_grid](1,m_i0,m_j0,m_k0);
      float_sw4 eta2 = eta[m_grid](2,m_i0,m_j0,m_k0);
      float_sw4 eta3 = eta[m_grid](3,m_i0,m_j0,m_k0);
      float_sw4 h3   = h[m_grid]*h[m_grid]*h[m_grid];

      if( 1 <= m_k0 && m_k0 <= 4 )
	 h3 *= normwgh[m_k0-1];

      float_sw4 c1 = g*h3;
      float_sw4 c2 = g0*dt2o12*h3;

      // (pos,pos)
      for( int m=0 ; m < 3 ; m++ )
	 for( int j=m ; j < 3 ; j++ )
	 {
	    hessian[m+11*j] -= (kap1*c1+eta1*c2)*m_hesspos1[m+3*j]+(kap2*c1+eta2*c2)*m_hesspos2[m+3*j]+
	       (kap3*c1+eta3*c2)*m_hesspos3[m+3*j];
	 }
      // (pos,mij)
      for( int m = 0; m < 3 ; m++ )
      {
	 int j=3;
	 hessian[m+11*j] -= (c1*kap1+eta1*c2)*m_dddp[m];
	 j = 4;
	 hessian[m+11*j] -= (c1*kap1+eta1*c2)*m_dddp[m+3] + (c1*kap2+eta2*c2)*m_dddp[m];
	 j = 5;
	 hessian[m+11*j] -= (c1*kap1+eta1*c2)*m_dddp[m+6] + (c1*kap3+eta3*c2)*m_dddp[m];      
	 j = 6;
	 hessian[m+11*j] -= (c1*kap2+eta2*c2)*m_dddp[m+3];
	 j = 7;
	 hessian[m+11*j] -= (c1*kap2+eta2*c2)*m_dddp[m+6] + (c1*kap3+eta3*c2)*m_dddp[m+3];
	 j = 8;
	 hessian[m+11*j] -= (c1*kap3+eta3*c2)*m_dddp[m+6];
      }
      // (pos,t0)
      float_sw4 dgt0 = -C6SmoothBump_t(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      float_sw4 dgom =  C6SmoothBump_om(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);

      float_sw4 c2t0  = dgt0*dt2o12;
      float_sw4 c2om0 = dgom*dt2o12;

      dgt0 = dgt0 - dt2o12*C6SmoothBump_ttt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      dgom = dgom + dt2o12*C6SmoothBump_omtt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      float_sw4 c1t0  = dgt0;
      float_sw4 c1om0 = dgom;

      //define c1t0,c2t0, c1om0, c2om0
      for( int m=0 ; m < 9 ; m++ )
      {
	 int j = 9;
	 hessian[m+11*j] -= ((c1t0*kap1+c2t0*eta1)*m_jacobian[3*m] + (c1t0*kap2+c2t0*eta2)*m_jacobian[3*m+1] +
			     (c1t0*kap3+c2t0*eta3)*m_jacobian[3*m+2])*h3;
	 j = 10;
	 hessian[m+11*j] -= ((c1om0*kap1+c2om0*eta1)*m_jacobian[3*m] + (c1om0*kap2+c2om0*eta2)*m_jacobian[3*m+1] +
			     (c1om0*kap3+c2om0*eta3)*m_jacobian[3*m+2])*h3;
      }
      float_sw4 cmfact0 = (kap1*mForces[0]+kap2*mForces[1]+kap3*mForces[2]);
      float_sw4 cmfact  = ((kap1*mForces[0]+kap2*mForces[1]+kap3*mForces[2])+
		      dt2o12*(eta1*mForces[0]+eta2*mForces[1]+eta3*mForces[2]));
      // Second derivatives of time function
      float_sw4 d2gdt02   =  C6SmoothBump_tt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      float_sw4 d2gdt0dom = -Gaussian_tom(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      float_sw4 d2gdomdom =  Gaussian_omom(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);

      float_sw4 dgdttt0t0  =  dt2o12*Gaussian_tttt(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      float_sw4 dgdttt0om  = -dt2o12*Gaussian_tttom(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);
      float_sw4 dgdttomom  =  dt2o12*Gaussian_ttomom(mFreq,t-mT0,mPar, mNpar, mIpar, mNipar);

      int m = 9;
      int j = 9;
      hessian[m+11*j] -= (cmfact*d2gdt02 + cmfact0*dgdttt0t0)*h3;
      j = 10;
      hessian[m+11*j] -= (cmfact*d2gdt0dom + cmfact0*dgdttt0om)*h3;
      m = 10;
      j = 10;
      hessian[m+11*j] -= (cmfact*d2gdomdom + cmfact0*dgdttomom)*h3;
   }
}

//-----------------------------------------------------------------------
void GridPointSource::print_info() const
{
   cout << "-----------------------------------------------------------------------"<<endl;
   cout << " position " << m_i0 << " " << m_j0 << " " << m_k0 << endl;
   cout << "Forces = " << mForces[0] << " " << mForces[1] << " " << mForces[2] << endl;
   cout << " jac = \n";
   for( int c=0 ; c < 3 ; c++ )
   {
       for( int m=0 ; m < 9 ; m++ )
	  cout << m_jacobian[c+3*m] << " " ;
       cout << endl;
   }
   cout << "Time dep " << mTimeDependence << endl;
   cout << "-----------------------------------------------------------------------"<<endl;

}

//-----------------------------------------------------------------------
void GridPointSource::set_sort_key( size_t key )
{
   m_key = key;
}

//-----------------------------------------------------------------------
SYCL_EXTERNAL void GridPointSource::init_dev()
{
   mPar = mdevPar;
   mIpar = mdevIpar;
}
