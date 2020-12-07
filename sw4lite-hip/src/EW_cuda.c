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
#include "EW.h"
   
#include "EWCuda.h"

#include "device-routines.h"

  
//-----------------------------------------------------------------------
void EW::evalRHSCU(vector<Sarray> & a_U, vector<Sarray>& a_Mu, vector<Sarray>& a_Lambda,
		   vector<Sarray> & a_Uacc, int st )
{
   dim3 gridsize, blocksize;
   gridsize.x  = m_gpu_gridsize[0];
   gridsize.y  = m_gpu_gridsize[1];
   gridsize.z  = m_gpu_gridsize[2];
   blocksize.x = m_gpu_blocksize[0];
   blocksize.y = m_gpu_blocksize[1];
   blocksize.z = m_gpu_blocksize[2];
   for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
   {
      if( m_corder ) 
      {
        int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
        int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
        int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
        dim3 block(RHS4_BLOCKX,RHS4_BLOCKY);
        dim3 grid;
        grid.x = (ni + block.x - 1) / block.x;
        grid.y = (nj + block.y - 1) / block.y;
        grid.z = 1;
        hipLaunchKernelGGL(rhs4center_dev_rev_v2, dim3(grid), dim3(block), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],
            m_kEnd[g], a_Uacc[g].dev_ptr(), a_U[g].dev_ptr(), a_Mu[g].dev_ptr(),
            a_Lambda[g].dev_ptr(), mGridSize[g],
            dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g], m_ghost_points );
      }
      else
      {
        int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
        int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
        int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
        dim3 block(RHS4_BLOCKX,RHS4_BLOCKY);
        dim3 grid;
        grid.x = (ni + block.x - 1) / block.x;
        grid.y = (nj + block.y - 1) / block.y;
        grid.z = 1;
        hipLaunchKernelGGL(rhs4center_dev_v2, dim3(grid), dim3(block), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],
	   m_kEnd[g], a_Uacc[g].dev_ptr(), a_U[g].dev_ptr(), a_Mu[g].dev_ptr(),
	   a_Lambda[g].dev_ptr(), mGridSize[g],
	   dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g], m_ghost_points );
      }
   }
// Boundary operator at upper boundary
   blocksize.z = 1;
   gridsize.z  = 6;

   for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
   {
      if( m_onesided[g][4] )
      {
	 if( m_corder )
	    hipLaunchKernelGGL(rhs4upper_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],  m_kEnd[g],
	      a_Uacc[g].dev_ptr(), a_U[g].dev_ptr(), a_Mu[g].dev_ptr(),
	      a_Lambda[g].dev_ptr(), mGridSize[g],
	      dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g], m_ghost_points );
	 else
	    hipLaunchKernelGGL(rhs4upper_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],  m_kEnd[g],
	      a_Uacc[g].dev_ptr(), a_U[g].dev_ptr(), a_Mu[g].dev_ptr(),
	      a_Lambda[g].dev_ptr(), mGridSize[g],
	      dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g], m_ghost_points );
      }
   }

   if( m_topography_exists )
   {
      gridsize.x  = m_gpu_gridsize[0];
      gridsize.y  = m_gpu_gridsize[1];
      gridsize.z  = m_gpu_gridsize[2];
      blocksize.x = m_gpu_blocksize[0];
      blocksize.y = m_gpu_blocksize[1];
      blocksize.z = m_gpu_blocksize[2];
      int g=mNumberOfGrids-1;
   // Boundary operator at upper boundary
      int onesided4 = 0;  
      if( m_onesided[g][4] )
      {
        onesided4 = 1;
        blocksize.z = 1;
        gridsize.z  = 6;
        if( m_corder )
  	  hipLaunchKernelGGL(rhs4sgcurvupper_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],
	          m_kEnd[g], a_U[g].dev_ptr(), a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(),
	          mMetric.dev_ptr(), mJ.dev_ptr(), a_Uacc[g].dev_ptr(), 
	          dev_sg_str_x[g], dev_sg_str_y[g], m_ghost_points );
        else
	  hipLaunchKernelGGL(rhs4sgcurvupper_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],
	          m_kEnd[g], a_U[g].dev_ptr(), a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(),
	          mMetric.dev_ptr(), mJ.dev_ptr(), a_Uacc[g].dev_ptr(), 
	          dev_sg_str_x[g], dev_sg_str_y[g], m_ghost_points );
      }
      gridsize.z  = m_gpu_gridsize[2];
      blocksize.z = m_gpu_blocksize[2];
      if( m_corder )
      {
        int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
        int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
        int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
        dim3 block(RHS4_BLOCKX,RHS4_BLOCKY);
        dim3 grid;
        grid.x = (ni + block.x - 1) / block.x;
        grid.y = (nj + block.y - 1) / block.y;
        grid.z = 1;
	hipLaunchKernelGGL(rhs4sgcurv_dev_rev_v2, dim3(grid), dim3(block), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],
	        m_kEnd[g], a_U[g].dev_ptr(), a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(),
	        mMetric.dev_ptr(), mJ.dev_ptr(), a_Uacc[g].dev_ptr(), 
	        onesided4, dev_sg_str_x[g], dev_sg_str_y[g], m_ghost_points );
      }
      else
	hipLaunchKernelGGL(rhs4sgcurv_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g],
	        m_kEnd[g], a_U[g].dev_ptr(), a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(),
	        mMetric.dev_ptr(), mJ.dev_ptr(), a_Uacc[g].dev_ptr(), 
	        onesided4, dev_sg_str_x[g], dev_sg_str_y[g], m_ghost_points );
   }
}

//-----------------------------------------------------------------------
void EW::evalPredictorCU( vector<Sarray> & a_Up, vector<Sarray> & a_U, vector<Sarray> & a_Um,
			  vector<Sarray>& a_Rho, vector<Sarray> & a_Lu, vector<Sarray> & a_F, int st )
{
   float_sw4 dt2 = mDt*mDt;
   dim3 gridsize, blocksize;
   gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;
   for( int g=0 ; g<mNumberOfGrids; g++ )
   {
      if( m_corder )
	 hipLaunchKernelGGL(pred_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
								   m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
								   a_Um[g].dev_ptr(), a_Lu[g].dev_ptr(), a_F[g].dev_ptr(),
								   a_Rho[g].dev_ptr(), dt2, m_ghost_points );
      else
	 hipLaunchKernelGGL(pred_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
								   m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
								   a_Um[g].dev_ptr(), a_Lu[g].dev_ptr(), a_F[g].dev_ptr(),
								   a_Rho[g].dev_ptr(), dt2, m_ghost_points );
   }
}

//---------------------------------------------------------------------------
void EW::evalCorrectorCU( vector<Sarray> & a_Up, vector<Sarray>& a_Rho,
			  vector<Sarray> & a_Lu, vector<Sarray> & a_F, int st )
{
   float_sw4 dt4 = mDt*mDt*mDt*mDt;  
   dim3 gridsize, blocksize;
   gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;
   for( int g=0 ; g<mNumberOfGrids; g++ )
   {
      if( m_corder )
	 hipLaunchKernelGGL(corr_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
								m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(), a_Lu[g].dev_ptr(),
								a_F[g].dev_ptr(), a_Rho[g].dev_ptr(), dt4,
								m_ghost_points );
      else
	 hipLaunchKernelGGL(corr_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
								m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(), a_Lu[g].dev_ptr(),
								a_F[g].dev_ptr(), a_Rho[g].dev_ptr(), dt4,
								m_ghost_points );
   }
}

//---------------------------------------------------------------------------
void EW::evalDpDmInTimeCU(vector<Sarray> & a_Up, vector<Sarray> & a_U, vector<Sarray> & a_Um,
			  vector<Sarray> & a_Uacc, int st )
{
   float_sw4 dt2i = 1./(mDt*mDt);
   dim3 gridsize, blocksize;
   gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;
   for(int g=0 ; g<mNumberOfGrids; g++ )
   {
      hipLaunchKernelGGL(dpdmt_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
								 m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
								 a_Um[g].dev_ptr(), a_Uacc[g].dev_ptr(), dt2i,
								 m_ghost_points );
   }
}

//-----------------------------------------------------------------------
void EW::addSuperGridDampingCU(vector<Sarray> & a_Up, vector<Sarray> & a_U,
			     vector<Sarray> & a_Um, vector<Sarray> & a_Rho, int st )
{
   dim3 gridsize, blocksize;
   gridsize.x  = m_gpu_gridsize[0];
   gridsize.y  = m_gpu_gridsize[1];
   gridsize.z  = m_gpu_gridsize[2];
   blocksize.x = m_gpu_blocksize[0];
   blocksize.y = m_gpu_blocksize[1];
   blocksize.z = m_gpu_blocksize[2];
   for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
   {
      if( m_sg_damping_order == 4 )
      {
	 if( m_corder )
         {
           int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
           int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
           int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
           const dim3 block(ADDSGD4_BLOCKX,ADDSGD4_BLOCKY,1);
           dim3 grid;
           grid.x = (ni + block.x - 1) / block.x;
           grid.y = (nj + block.y - 1) / block.y;
           grid.z = 1; 
           hipLaunchKernelGGL(addsgd4_dev_rev_v2, dim3(grid), dim3(block), 0, m_cuobj->m_stream[st], 
             m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
             m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(),
             a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
             dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
             dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
             dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
             m_supergrid_damping_coefficient, m_ghost_points );
         }
	 else
         {
           int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
           int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
           int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
           const dim3 block(ADDSGD4_BLOCKX,ADDSGD4_BLOCKY,1);
           dim3 grid;
           grid.x = (ni + block.x - 1) / block.x;
           grid.y = (nj + block.y - 1) / block.y;
           grid.z = 1; 
           hipLaunchKernelGGL(addsgd4_dev_v2, dim3(grid), dim3(block), 0, m_cuobj->m_stream[st], 
             m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
             m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(),
             a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
             dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
             dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
             dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
             m_supergrid_damping_coefficient, m_ghost_points );
          }
      }
      else if(  m_sg_damping_order == 6 )
      {
	 if( m_corder )
	    hipLaunchKernelGGL(addsgd6_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
								      m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(), 
                                                                      a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
								      dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
								      dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
								      dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
								      m_supergrid_damping_coefficient, m_ghost_points );
	 else
	    hipLaunchKernelGGL(addsgd6_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
								      m_kStart[g], m_kEnd[g], a_Up[g].dev_ptr(), 
                                                                      a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
								      dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
								      dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
								      dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
								      m_supergrid_damping_coefficient, m_ghost_points );
      }
   }

   if( m_topography_exists )
   {
      int g=mNumberOfGrids-1;
      if( m_sg_damping_order == 4 )
      {
	 if( m_corder )
            hipLaunchKernelGGL(addsgd4c_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st], 
              m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g], m_kEnd[g], 
              a_Up[g].dev_ptr(), a_U[g].dev_ptr(), 
              a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
              dev_sg_dc_x[g], dev_sg_dc_y[g], 
              dev_sg_str_x[g], dev_sg_str_y[g], 
              mJ.dev_ptr(), dev_sg_corner_x[g], dev_sg_corner_y[g], 
              m_supergrid_damping_coefficient, m_ghost_points );
	 else
            hipLaunchKernelGGL(addsgd4c_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st], 
              m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g], m_kEnd[g], 
              a_Up[g].dev_ptr(), a_U[g].dev_ptr(), 
              a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
              dev_sg_dc_x[g], dev_sg_dc_y[g], 
              dev_sg_str_x[g], dev_sg_str_y[g], 
              mJ.dev_ptr(), dev_sg_corner_x[g], dev_sg_corner_y[g], 
              m_supergrid_damping_coefficient, m_ghost_points );
      }
      else if(  m_sg_damping_order == 6 )
      {
	 if( m_corder )
	    hipLaunchKernelGGL(addsgd6c_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  
              m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g], m_kEnd[g], 
              a_Up[g].dev_ptr(), a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
	      dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_str_x[g], dev_sg_str_y[g], 
	      mJ.dev_ptr(), dev_sg_corner_x[g], dev_sg_corner_y[g], 
              m_supergrid_damping_coefficient, m_ghost_points );
	 else
	    hipLaunchKernelGGL(addsgd6c_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  
              m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g], m_kEnd[g], 
              a_Up[g].dev_ptr(), a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
	      dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_str_x[g], dev_sg_str_y[g], 
	      mJ.dev_ptr(), dev_sg_corner_x[g], dev_sg_corner_y[g], 
              m_supergrid_damping_coefficient, m_ghost_points );
      }
   }
}

//-----------------------------------------------------------------------
void EW::setupSBPCoeff()
{
   //   float_sw4 gh2; // this coefficient is also stored in m_ghcof[0]
   if (mVerbose >=1 && m_myrank == 0)
      cout << "Setting up SBP boundary stencils" << endl;
// get coefficients for difference approximation of 2nd derivative with variable coefficients
   GetStencilCoefficients( m_acof, m_ghcof, m_bope, m_sbop );
   copy_stencilcoefficients1( m_acof, m_ghcof, m_bope, m_sbop );
}

//-----------------------------------------------------------------------
void EW::copy_supergrid_arrays_to_device()
{
  dev_sg_str_x.resize(mNumberOfGrids);
  dev_sg_str_y.resize(mNumberOfGrids);
  dev_sg_str_z.resize(mNumberOfGrids);
  dev_sg_dc_x.resize(mNumberOfGrids);
  dev_sg_dc_y.resize(mNumberOfGrids);
  dev_sg_dc_z.resize(mNumberOfGrids);
  dev_sg_corner_x.resize(mNumberOfGrids);
  dev_sg_corner_y.resize(mNumberOfGrids);
  dev_sg_corner_z.resize(mNumberOfGrids);
  if( m_ndevice > 0 )
  {
     hipError_t retcode;
     for( int g=0 ; g<mNumberOfGrids; g++) 
     {
	// sg_str
	retcode = hipMalloc( (void**)&dev_sg_str_x[g], sizeof(float_sw4)*(m_iEnd[g]-m_iStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc x returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMalloc( (void**)&dev_sg_str_y[g], sizeof(float_sw4)*(m_jEnd[g]-m_jStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc y returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMalloc( (void**)&dev_sg_str_z[g], sizeof(float_sw4)*(m_kEnd[g]-m_kStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc z returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_str_x[g], m_sg_str_x[g], sizeof(float_sw4)*(m_iEnd[g]-m_iStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy x returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_str_y[g], m_sg_str_y[g], sizeof(float_sw4)*(m_jEnd[g]-m_jStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy y returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_str_z[g], m_sg_str_z[g], sizeof(float_sw4)*(m_kEnd[g]-m_kStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy z returned "
		<< hipGetErrorString(retcode) << endl;

	// sg_dc
	retcode = hipMalloc( (void**)&dev_sg_dc_x[g], sizeof(float_sw4)*(m_iEnd[g]-m_iStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc dc_x returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMalloc( (void**)&dev_sg_dc_y[g], sizeof(float_sw4)*(m_jEnd[g]-m_jStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc dc_y returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMalloc( (void**)&dev_sg_dc_z[g], sizeof(float_sw4)*(m_kEnd[g]-m_kStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc dc_z returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_dc_x[g], m_sg_dc_x[g], sizeof(float_sw4)*(m_iEnd[g]-m_iStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy dc_x returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_dc_y[g], m_sg_dc_y[g], sizeof(float_sw4)*(m_jEnd[g]-m_jStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy dc_y returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_dc_z[g], m_sg_dc_z[g], sizeof(float_sw4)*(m_kEnd[g]-m_kStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy dc_z returned "
		<< hipGetErrorString(retcode) << endl;
	// sg_corner
	retcode = hipMalloc( (void**)&dev_sg_corner_x[g], sizeof(float_sw4)*(m_iEnd[g]-m_iStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc corner_x returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMalloc( (void**)&dev_sg_corner_y[g], sizeof(float_sw4)*(m_jEnd[g]-m_jStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc corner_y returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMalloc( (void**)&dev_sg_corner_z[g], sizeof(float_sw4)*(m_kEnd[g]-m_kStart[g]+1));
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMalloc corner_z returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_corner_x[g], m_sg_corner_x[g], sizeof(float_sw4)*(m_iEnd[g]-m_iStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy corner_x returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_corner_y[g], m_sg_corner_y[g], sizeof(float_sw4)*(m_jEnd[g]-m_jStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy corner_y returned "
		<< hipGetErrorString(retcode) << endl;
	retcode = hipMemcpy( dev_sg_corner_z[g], m_sg_corner_z[g], sizeof(float_sw4)*(m_kEnd[g]-m_kStart[g]+1),
			      hipMemcpyHostToDevice );
	if( retcode != hipSuccess )
	   cout << "Error, EW::copy_supergrid_arrays_to_device hipMemcpy corner_z returned "
		<< hipGetErrorString(retcode) << endl;
     }
  }
}

//-----------------------------------------------------------------------
void EW::copy_material_to_device()
{
   for( int g=0 ; g<mNumberOfGrids; g++)
   {
      mMu[g].copy_to_device( m_cuobj );
      mLambda[g].copy_to_device( m_cuobj );
      mRho[g].copy_to_device( m_cuobj );
   }
   mJ.copy_to_device(m_cuobj);
   mMetric.copy_to_device(m_cuobj);
}

//-----------------------------------------------------------------------
void EW::find_cuda_device()
{
   hipError_t retcode;
   hipDeviceProp_t prop;
   retcode = hipGetDeviceCount(&m_ndevice);
   if( retcode != hipSuccess )
   {
      cout << "Error from hipGetDeviceCount: Error string = " <<
	 hipGetErrorString(retcode) << endl;
   }
   // Note: This will not work if some nodes have GPU and others do not
   // It is assumed that all nodes are identical wrt GPU
   if( m_ndevice > 0 && m_myrank == 0 )
   {
      cout << m_ndevice << " Cuda devices found:" << endl;
      for( int i=0 ;  i < m_ndevice ; i++ )
      {
	 retcode = hipGetDeviceProperties( &prop, i );
	 cout << "      Device " << i << ": name = " << prop.name <<
	    ",  Compute capability:" << prop.major << "." << prop.minor << 
	    ",  Memory (GB) " << (prop.totalGlobalMem  >> 30) << endl;
      }
   }
   /*Added following line for all ranks 
   retcode = hipGetDeviceProperties(&prop, 0 );

   // Check block size
   CHECK_INPUT( m_gpu_blocksize[0] <= prop.maxThreadsDim[0],
		"Error: max block x " << m_gpu_blocksize[0] << " too large\n");
   CHECK_INPUT( m_gpu_blocksize[1] <= prop.maxThreadsDim[1], 
		"Error: max block y " << m_gpu_blocksize[1] << " too large\n");
   CHECK_INPUT( m_gpu_blocksize[2] <= prop.maxThreadsDim[2],
		"Error: max block z " << m_gpu_blocksize[2] << " too large\n");
   CHECK_INPUT( m_gpu_blocksize[0]*m_gpu_blocksize[1]*m_gpu_blocksize[2] <= prop.maxThreadsPerBlock, 
   "Error: max number of threads per block " << prop.maxThreadsPerBlock <<
		" is exceeded \n");
		*/

   // Determine grid dimensions.
   int ghost = m_ghost_points;
   int ni=m_iEnd[0]-m_iStart[0]+1-2*ghost;
   int nj=m_jEnd[0]-m_jStart[0]+1-2*ghost;
   int nk=m_kEnd[0]-m_kStart[0]+1-2*ghost;
   bool m_gpu_overlap = false;
   if( m_gpu_overlap )
   {
      REQUIRE2( m_gpu_blocksize[0] > 2*ghost && m_gpu_blocksize[1] > 2*ghost 
	       && m_gpu_blocksize[2] > 2*ghost , "Error, need block size at least "
	       << 2*ghost+1 << " in each direction\n" );
      m_gpu_gridsize[0]=ni/(m_gpu_blocksize[0]-2*ghost);
      if( ni % (m_gpu_blocksize[0]-2*ghost)  != 0 )
	 m_gpu_gridsize[0]++;
      m_gpu_gridsize[1]=nj/(m_gpu_blocksize[1]-2*ghost);
      if( nj % (m_gpu_blocksize[1]-2*ghost)  != 0 )
	 m_gpu_gridsize[1]++;
      m_gpu_gridsize[2]=nk/(m_gpu_blocksize[2]-2*ghost);
      if( nk % (m_gpu_blocksize[2]-2*ghost)  != 0 )
	 m_gpu_gridsize[2]++;
   }
   else
   {
      m_gpu_gridsize[0]=ni/m_gpu_blocksize[0];
      if( ni % m_gpu_blocksize[0]  != 0 )
	 m_gpu_gridsize[0]++;
      m_gpu_gridsize[1]=nj/m_gpu_blocksize[1];
      if( nj % m_gpu_blocksize[1]  != 0 )
	 m_gpu_gridsize[1]++;
      m_gpu_gridsize[2]=nk/m_gpu_blocksize[2];
      if( nk % m_gpu_blocksize[2]  != 0 )
	 m_gpu_gridsize[2]++;
   }
   if( m_myrank == 0 )
   {
   cout << " size of domain " << ni << " x " << nj << " x " << nk << " grid points (excluding ghost points)\n";
   cout << " GPU block size " <<  m_gpu_blocksize[0] << " x " <<  m_gpu_blocksize[1] << " x " <<  m_gpu_blocksize[2] << "\n";
   cout << " GPU grid size  " <<  m_gpu_gridsize[0]  << " x " <<  m_gpu_gridsize[1]  << " x " <<  m_gpu_gridsize[2] << endl;
   cout << " GPU cores " << m_gpu_blocksize[0]*m_gpu_gridsize[0] << " x " 
	   << m_gpu_blocksize[1]*m_gpu_gridsize[1] << " x "
	   << m_gpu_blocksize[2]*m_gpu_gridsize[2] << endl;
   }
   if( m_ndevice > 0 )
   {
      // create two streams
      m_cuobj = new EWCuda( m_ndevice, 2 );
   }


   if( m_ndevice == 0 && m_myrank == 0 )
      cout << " no GPUs found" << endl;
   if( m_ndevice == 0 )
      m_cuobj = new EWCuda(0,0);
}

//-----------------------------------------------------------------------
bool EW::check_for_nan_GPU( vector<Sarray>& a_U, int verbose, string name )
{
   dim3 gridsize, blocksize;
   gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;

   int  *retval_dev, retval_host = 0;
   int  *idx_dev, idx_host = 0;
   int cnan, inan, jnan, knan;
   int  nijk, nij, ni;

   hipMalloc( (void **)&retval_dev, sizeof(int) );
   hipMemcpy( retval_dev, &retval_host, sizeof(int), hipMemcpyHostToDevice );
   hipMalloc( (void **)&idx_dev, sizeof (int) );
   hipMemcpy( idx_dev, &idx_host, sizeof(int), hipMemcpyHostToDevice );

   for( int g=0 ; g<mNumberOfGrids; g++ )
   {
      hipLaunchKernelGGL(check_nan_dev, dim3(gridsize), dim3(blocksize), 0, 0,  m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g],
					     m_kStart[g], m_kEnd[g], a_U[g].dev_ptr(), retval_dev, idx_dev );

      hipMemcpy( &retval_host, retval_dev, sizeof(int), hipMemcpyDeviceToHost );

      if ( retval_host != 0) 
      {
         hipMemcpy(&idx_host, idx_dev, sizeof(int), hipMemcpyDeviceToHost);

         nijk = (m_kEnd[g]-m_kStart[g]+1)*(m_jEnd[g]-m_jStart[g]+1)*(m_iEnd[g]-m_iStart[g]+1);
         nij  = (m_jEnd[g]-m_jStart[g]+1)*(m_iEnd[g]-m_iStart[g]+1);
         ni   = m_iEnd[g]-m_iStart[g]+1;

         cnan = idx_host/nijk;
         idx_host = idx_host - cnan*nijk; 
         knan = idx_host / nij; 
         idx_host = idx_host - knan*nij;
         jnan = idx_host/ni;
         inan = idx_host - jnan*ni; 

         cout << "grid " << g << " array " << name << " found " << retval_host << " nans. First nan at " <<
                    cnan << " " << inan << " " << jnan << " " << knan << endl;

         return false; 
      }
   }

  return true;
}

//-----------------------------------------------------------------------
void EW::ForceCU( float_sw4 t, Sarray* dev_F, bool tt, int st )
{
   dim3 blocksize, gridsize;
   gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;
   hipLaunchKernelGGL(forcing_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  t, dev_F, mNumberOfGrids, dev_point_sources, m_point_sources.size(),
					dev_identsources, m_identsources.size(), tt );
}

//-----------------------------------------------------------------------
void EW::init_point_sourcesCU( )
{
   dim3 blocksize, gridsize;
   gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;
   hipLaunchKernelGGL(init_forcing_dev, dim3(gridsize), dim3(blocksize), 0, 0,  dev_point_sources, m_point_sources.size() );
}

void EW::cartesian_bc_forcingCU( float_sw4 t, vector<float_sw4**> & a_BCForcing,
                                 vector<Source*>& a_sources , int st)
// assign the boundary forcing arrays dev_BCForcing[g][side]
{
  hipError_t retcode;
  for(int g=0 ; g<mNumberOfGrids; g++ )
  {
    if( m_point_source_test )
    {
      for( int side=0 ; side < 6 ; side++ )
      {
        size_t nBytes = sizeof(float_sw4)*3*m_NumberOfBCPoints[g][side];
        if( m_bcType[g][side] == bDirichlet )
        {
          get_exact_point_source( a_BCForcing[g][side], t, g, *a_sources[0], &m_BndryWindow[g][6*side] );
          retcode = hipMemcpyAsync( dev_BCForcing[g][side], a_BCForcing[g][side], nBytes, hipMemcpyHostToDevice, m_cuobj->m_stream[st]);
          if( retcode != hipSuccess )
            cout << "Error, EW::cartesian_bc_forcing_CU hipMemcpy x returned " << hipGetErrorString(retcode) << endl;
        }
        else
        {
          hipMemsetAsync( dev_BCForcing[g][side], 0, nBytes , m_cuobj->m_stream[st]);
        }
      }
    }
    else
    {
      for( int side=0 ; side < 6 ; side++ )
      {
        size_t nBytes = sizeof(float_sw4)*3*m_NumberOfBCPoints[g][side];
        hipMemsetAsync( dev_BCForcing[g][side], 0, nBytes , m_cuobj->m_stream[st]);
      }
    }
  }
}

//-----------------------------------------------------------------------

void EW::copy_bcforcing_arrays_to_device()
{

   //Set up boundary data array on the deivec
  if(m_ndevice > 0 )
  {
    hipError_t retcode;
    dev_BCForcing.resize(mNumberOfGrids);
    for( int g = 0; g <mNumberOfGrids; g++ )
    {
      dev_BCForcing[g] = new float_sw4*[6];
      for (int side=0; side < 6; side++)
      {
        dev_BCForcing[g][side] = NULL;
        if (m_bcType[g][side] == bStressFree || m_bcType[g][side] == bDirichlet || m_bcType[g][side] == bSuperGrid)
        {
          size_t nBytes = sizeof(float_sw4)*3*m_NumberOfBCPoints[g][side];
          retcode  = hipMalloc((void**) &dev_BCForcing[g][side], nBytes );
          if( retcode != hipSuccess )
          {
             cout << "Error, EW::copy_bcforcing_arrays_to_device hipMalloc x returned " << hipGetErrorString(retcode) << endl;
             exit(-1);
          }
          retcode = hipMemcpy( dev_BCForcing[g][side], BCForcing[g][side], nBytes, hipMemcpyHostToDevice );
          if( retcode != hipSuccess )
          {
            cout << "Error, EW::copy_bcforcing_arrays_to_device hipMemcpy x returned " << hipGetErrorString(retcode) << endl;
            exit(-1);
          }
        }
      }
    }
  }
}

//-----------------------------------------------------------------------

void EW::copy_bctype_arrays_to_device()
{
  // Set up boundary type array on the deivec
  if(m_ndevice > 0 )
  {
    hipError_t retcode;
    dev_bcType.resize(mNumberOfGrids);
    for( int g = 0; g <mNumberOfGrids; g++ )
    {
      size_t nBytes = sizeof(boundaryConditionType)*6;
      retcode = hipMalloc( (void**) &dev_bcType[g], nBytes );
      if( retcode != hipSuccess )
        cout << "Error, EW::copy_bctype_arrays_to_device hipMalloc x returned " << hipGetErrorString(retcode) << endl;
      retcode = hipMemcpy( dev_bcType[g], m_bcType[g], nBytes, hipMemcpyHostToDevice );
      if( retcode != hipSuccess )
        cout << "Error, EW::copy_bctype_arrays_to_device hipMemcpy x returned " << hipGetErrorString(retcode) << endl;
    }
  }
}

//-----------------------------------------------------------------------

void EW::copy_bndrywindow_arrays_to_device()
{
  //Set up boundary window array on the deivec
  if(m_ndevice > 0 )
  {
    hipError_t retcode;
    dev_BndryWindow.resize(mNumberOfGrids);
    for( int g = 0; g <mNumberOfGrids; g++ )
    {
      size_t nBytes = sizeof(int)*36;
      retcode = hipMalloc( (void**) &dev_BndryWindow[g], nBytes );
      if( retcode != hipSuccess )
        cout << "Error, EW::copy_bndrywindow_arrays_to_device hipMalloc x returned " << hipGetErrorString(retcode) << endl;
      retcode = hipMemcpy( dev_BndryWindow[g], m_BndryWindow[g], nBytes, hipMemcpyHostToDevice );
      if( retcode != hipSuccess )
        cout << "Error, EW::copy_bndrywindow_arrays_to_device hipMemcpy x returned " << hipGetErrorString(retcode) << endl;
    }
  }
}

// New functions from Guillaume. 

//---------------------------------------------------------------------------
void EW::RHSPredCU_center(vector<Sarray> & a_Up, vector<Sarray> & a_U, vector<Sarray> & a_Um,
                          vector<Sarray>& a_Mu, vector<Sarray>& a_Lambda,
                          vector<Sarray>& a_Rho, vector<Sarray>& a_F, int st) {
  int ni, nj, nk, startk, endk;

  for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
    {
      // Cube leading dimensions
      ni = m_iEnd[g] - m_iStart[g] + 1;
      nj = m_jEnd[g] - m_jStart[g] + 1;
      nk = m_kEnd[g] - m_kStart[g] + 1;
      
      // If there's a free surface, start at k=8 instead of 2,
      // the free surface will compute k=[2:7]
      if( m_onesided[g][4] )
        startk = 8;
      else
        startk = 2;
      if( m_onesided[g][5] )
        endk = m_global_nz[g] - 7;
      else
        endk = nk - 3;

      // RHS and predictor in center
      rhs4_pred_gpu (4, ni-5, 4, nj-5, startk, endk,
                     ni, nj, nk,
                     a_Up[g].dev_ptr(), a_U[g].dev_ptr(), a_Um[g].dev_ptr(),
                     a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                     dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                     mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);

    }  
}  

//---------------------------------------------------------------------------
void EW::RHSPredCU_boundary(vector<Sarray> & a_Up, vector<Sarray> & a_U, vector<Sarray> & a_Um,
                                  vector<Sarray>& a_Mu, vector<Sarray>& a_Lambda,
                                  vector<Sarray>& a_Rho, vector<Sarray>& a_F, int st) {
  int ni, nj, nk, nz, startk;

  for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
    {
      // Cube leading dimensions
      ni = m_iEnd[g] - m_iStart[g] + 1;
      nj = m_jEnd[g] - m_jStart[g] + 1;
      nk = m_kEnd[g] - m_kStart[g] + 1;
     
      nz = m_global_nz[g];  
      // If we have a free surface, the other kernels start at k=8 instead of 2,
      // the free surface will compute k=[2:7]
      if( m_onesided[g][4] )
        startk = 8;
      else
        startk = 2;

      // RHS and predictor in the X halos
      rhs4_X_pred_gpu (2, ni-3, 4, nj-5, startk, nk-3,
                       ni, nj, nk,
                       a_Up[g].dev_ptr(), a_U[g].dev_ptr(), a_Um[g].dev_ptr(),
                       a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                       dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                       mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);
		   
      // RHS and predictor in the Y halos
      rhs4_Y_pred_gpu (2, ni-3, 2, nj-3, startk, nk-3,
                       ni, nj, nk,
                       a_Up[g].dev_ptr(), a_U[g].dev_ptr(), a_Um[g].dev_ptr(),
                       a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                       dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                       mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);
  
      // Free surface and predictor
      if( m_onesided[g][4] )
        rhs4_lowk_pred_gpu (2, ni-3, 2, nj-3,
                            ni, nj, nk, nz,
                            a_Up[g].dev_ptr(), a_U[g].dev_ptr(), a_Um[g].dev_ptr(),
                            a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                            dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                            mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);

      if( m_onesided[g][5] )
        rhs4_highk_pred_gpu (2, ni-3, 2, nj-3,
                            ni, nj, nk, nz,
                            a_Up[g].dev_ptr(), a_U[g].dev_ptr(), a_Um[g].dev_ptr(),
                            a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                            dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                            mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);

    }  
}  
//---------------------------------------------------------------------------
void EW::RHSCorrCU_center(vector<Sarray> & a_Up, vector<Sarray> & a_U,
                         vector<Sarray>& a_Mu, vector<Sarray>& a_Lambda,
                         vector<Sarray>& a_Rho, vector<Sarray>& a_F, int st) {
  int ni, nj, nk, startk, endk;

  for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
    {
      // Cube leading dimensions
      ni = m_iEnd[g] - m_iStart[g] + 1;
      nj = m_jEnd[g] - m_jStart[g] + 1;
      nk = m_kEnd[g] - m_kStart[g] + 1;
      
      // If we have a free surface, the other kernels start at k=8 instead of 2,
      // the free surface will compute k=[2:7]
      if( m_onesided[g][4] )
        startk = 8;
      else
        startk = 2;
      if( m_onesided[g][5] )
        endk = m_global_nz[g] - 7;
      else
        endk = nk - 3;
  
      // RHS and corrector in the rest of the cube
      rhs4_corr_gpu (4, ni-5, 4, nj-5, startk, endk,
                     ni, nj, nk,
                     a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
                     a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                     dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                     mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);
    }  
}  

//---------------------------------------------------------------------------
void EW::RHSCorrCU_boundary(vector<Sarray> & a_Up, vector<Sarray> & a_U,
                         vector<Sarray>& a_Mu, vector<Sarray>& a_Lambda,
                         vector<Sarray>& a_Rho, vector<Sarray>& a_F, int st) {
  int ni, nj, nk, nz, startk;

  for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
    {
      // Cube leading dimensions
      ni = m_iEnd[g] - m_iStart[g] + 1;
      nj = m_jEnd[g] - m_jStart[g] + 1;
      nk = m_kEnd[g] - m_kStart[g] + 1;
      
      nz = m_global_nz[g];  
      // If we have a free surface, the other kernels start at k=8 instead of 2,
      // the free surface will compute k=[2:7]
      if( m_onesided[g][4] )
        startk = 8;
      else
        startk = 2;
  
      rhs4_X_corr_gpu (2, ni-3, 4, nj-5, startk, nk-3,
                     ni, nj, nk,
                     a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
                     a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                     dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                     mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);
      rhs4_Y_corr_gpu (2, ni-3, 2, nj-3, startk, nk-3,
                     ni, nj, nk,
                     a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
                     a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                     dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                     mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);
      // Free surface and corrector on low-k boundary
      if( m_onesided[g][4] )
        rhs4_lowk_corr_gpu (2, ni-3, 2, nj-3,
                            ni, nj, nk, nz,
                            a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
                            a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                            dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                            mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);

      if( m_onesided[g][5] )
        rhs4_highk_corr_gpu (2, ni-3, 2, nj-3,
                            ni, nj, nk, nz,
                            a_Up[g].dev_ptr(), a_U[g].dev_ptr(),
                            a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(), a_Rho[g].dev_ptr(), a_F[g].dev_ptr(),
                            dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                            mGridSize[g], mDt, m_corder, m_cuobj->m_stream[st]);

    }  
}  

//---------------------------------------------------------------------------
void EW::addSuperGridDampingCU_upper_boundary(vector<Sarray> & a_Up, vector<Sarray> & a_U,
                                              vector<Sarray> & a_Um, vector<Sarray> & a_Rho, int st )
{
  for(int g=0 ; g<mNumberOfGrids; g++ )
   {
     int ni = m_iEnd[g] - m_iStart[g] + 1;
     int nj = m_jEnd[g] - m_jStart[g] + 1;
     int nk = m_kEnd[g] - m_kStart[g] + 1;

     // If we have a free surface, the other kernels start at k=8 instead of 2,
     // the free surface will compute k=[2:7]
     int startk;
     if( m_onesided[g][4] )
       startk = 8;
     else
       startk = 2;

     if( m_sg_damping_order == 4 )
       {
         // X Halo (avoid the Y halos -> J=[4:NJ-5]
         addsgd4_X_gpu  (2, ni-3, 4, nj-5, startk, nk-3,
                         ni, nj, nk,
                         a_Up[g].dev_ptr(),  a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
                         dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
                         dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                         dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
                         m_supergrid_damping_coefficient, m_corder, m_cuobj->m_stream[st]);
         // Y halo
         addsgd4_Y_gpu  (2, ni-3, 2, nj-3, startk, nk-3,
                         ni, nj, nk,
                         a_Up[g].dev_ptr(),  a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
                         dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
                         dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                         dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
                         m_supergrid_damping_coefficient, m_corder, m_cuobj->m_stream[st]);
         
         // Free surface k=[2:7]
         if( m_onesided[g][4] )
           addsgd4_gpu  (2, ni-3, 2, nj-3, 2, 7,
                         ni, nj, nk,
                         a_Up[g].dev_ptr(),  a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
                         dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
                         dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                         dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
                         m_supergrid_damping_coefficient, m_corder, m_cuobj->m_stream[st]);
       }
   }
}

//---------------------------------------------------------------------------
void EW::addSuperGridDampingCU_center(vector<Sarray> & a_Up, vector<Sarray> & a_U,
                                      vector<Sarray> & a_Um, vector<Sarray> & a_Rho, int st )
{
  for(int g=0 ; g<mNumberOfGrids; g++ )
   {
     int ni = m_iEnd[g] - m_iStart[g] + 1;
     int nj = m_jEnd[g] - m_jStart[g] + 1;
     int nk = m_kEnd[g] - m_kStart[g] + 1;
     // If we have a free surface, the other kernels start at k=8 instead of 2,
     // the free surface will compute k=[2:7]
     int startk;
     if( m_onesided[g][4] )
       startk = 8;
     else
       startk = 2;
     if( m_sg_damping_order == 4 )
       {
         addsgd4_gpu( 4, ni-5, 4, nj-5, startk, nk-3,
                      ni, nj, nk,
                      a_Up[g].dev_ptr(),  a_U[g].dev_ptr(), a_Um[g].dev_ptr(), a_Rho[g].dev_ptr(),
                      dev_sg_dc_x[g], dev_sg_dc_y[g], dev_sg_dc_z[g],
                      dev_sg_str_x[g], dev_sg_str_y[g], dev_sg_str_z[g],
                      dev_sg_corner_x[g], dev_sg_corner_y[g], dev_sg_corner_z[g],
                      m_supergrid_damping_coefficient, m_corder, m_cuobj->m_stream[st]);
      }
   }
}


//-----------------------------------------------------------------------

void EW::pack_HaloArrayCU_X( Sarray& u, int g , int st)
{
  REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
	    << " nc = " << u.m_nc );
  int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
  MPI_Status status;
  dim3 gridsize, blocksize;
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize.x  = 1 * 1 * m_gpu_gridsize[2];
  gridsize.y  = 1;
  gridsize.z  = 1;
  blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize.y = 1;
  blocksize.z = 1;

  int ni = m_iEnd[g] - m_iStart[g] + 1;
  int nj = m_jEnd[g] - m_jStart[g] + 1;
  int nk = m_kEnd[g] - m_kStart[g] + 1;
  int n_m_ppadding1 = 3*nj*nk*m_ppadding;
  int n_m_ppadding2 = 3*ni*nk*m_ppadding;
  int idx_left = 0;
  int idx_right = n_m_ppadding2;
  int idx_up = 2*n_m_ppadding2;
  int idx_down = 2*n_m_ppadding2 + n_m_ppadding1;

  if( u.m_nc == 3 )
    {
      // X-direction communication
      if(m_corder)
        hipLaunchKernelGGL(BufferToHaloKernelY_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  &u(1,ie-(2*m_ppadding-1),jb,kb,true), &u(1,ib+m_ppadding,jb,kb,true),
            &dev_SideEdge_Send[g][idx_up], &dev_SideEdge_Send[g][idx_down],
            ni, nj, nk, m_ppadding, m_neighbor[0], m_neighbor[1], MPI_PROC_NULL );
      else
        hipLaunchKernelGGL(BufferToHaloKernelY_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  &u(1,ie-(2*m_ppadding-1),jb,kb,true), &u(1,ib+m_ppadding,jb,kb,true),
            &dev_SideEdge_Send[g][idx_up], &dev_SideEdge_Send[g][idx_down],
            ni, nj, nk, m_ppadding, m_neighbor[0],  m_neighbor[1], MPI_PROC_NULL );
      CheckCudaCall(hipGetLastError(), "BufferToHaloKernel<<<,>>>(...)", __FILE__, __LINE__);
    }
}
//-----------------------------------------------------------------------

void EW::pack_HaloArrayCU_Y( Sarray& u, int g , int st)
{
  REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
	    << " nc = " << u.m_nc );
  int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
  MPI_Status status;
  dim3 gridsize, blocksize;
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize.x  = 1 * 1 * m_gpu_gridsize[2];
  gridsize.y  = 1;
  gridsize.z  = 1;
  blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize.y = 1;
  blocksize.z = 1;

  int ni = m_iEnd[g] - m_iStart[g] + 1;
  int nj = m_jEnd[g] - m_jStart[g] + 1;
  int nk = m_kEnd[g] - m_kStart[g] + 1;
  int n_m_ppadding1 = 3*nj*nk*m_ppadding;
  int n_m_ppadding2 = 3*ni*nk*m_ppadding;
  int idx_left = 0;
  int idx_right = n_m_ppadding2;
  int idx_up = 2*n_m_ppadding2;
  int idx_down = 2*n_m_ppadding2 + n_m_ppadding1;

  if( u.m_nc == 3 )
    {
      // Y-direction communication	 
      if(m_corder)
        hipLaunchKernelGGL(BufferToHaloKernelX_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  &u(1,ib,jb+m_ppadding,kb,true), &u(1,ib,je-(2*m_ppadding-1),kb,true),
            &dev_SideEdge_Send[g][idx_left], &dev_SideEdge_Send[g][idx_right],
            ni, nj, nk, m_ppadding, m_neighbor[2], m_neighbor[3], MPI_PROC_NULL );
      else
        hipLaunchKernelGGL(BufferToHaloKernelX_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st] ,  &u(1,ib,jb+m_ppadding,kb,true), &u(1,ib,je-(2*m_ppadding-1),kb,true), 
            &dev_SideEdge_Send[g][idx_left], &dev_SideEdge_Send[g][idx_right],
            ni, nj, nk, m_ppadding, m_neighbor[2], m_neighbor[3], MPI_PROC_NULL );
      CheckCudaCall(hipGetLastError(), "BufferToHaloKernel<<<,>>>(...)", __FILE__, __LINE__);
      
    }
}
//-----------------------------------------------------------------------

void EW::unpack_HaloArrayCU_X( Sarray& u, int g , int st)
{
  REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
	    << " nc = " << u.m_nc );
  int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
  MPI_Status status;
  dim3 gridsize, blocksize;
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize.x  = 1 * 1 * m_gpu_gridsize[2];
  gridsize.y  = 1;
  gridsize.z  = 1;
  blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize.y = 1;
  blocksize.z = 1;

  int ni = m_iEnd[g] - m_iStart[g] + 1;
  int nj = m_jEnd[g] - m_jStart[g] + 1;
  int nk = m_kEnd[g] - m_kStart[g] + 1;
  int n_m_ppadding1 = 3*nj*nk*m_ppadding;
  int n_m_ppadding2 = 3*ni*nk*m_ppadding;
  int idx_left = 0;
  int idx_right = n_m_ppadding2;
  int idx_up = 2*n_m_ppadding2;
  int idx_down = 2*n_m_ppadding2 + n_m_ppadding1;

  if( u.m_nc == 3 )
    {
      // X-direction communication
      if(m_corder)
        hipLaunchKernelGGL(HaloToBufferKernelY_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  &u(1,ie-(m_ppadding-1),jb,kb,true), &u(1,ib,jb,kb,true),
            &dev_SideEdge_Recv[g][idx_up], &dev_SideEdge_Recv[g][idx_down], ni, nj, nk, m_ppadding,
            m_neighbor[0], m_neighbor[1], MPI_PROC_NULL );
      else
        hipLaunchKernelGGL(HaloToBufferKernelY_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st] ,  &u(1,ie-(m_ppadding-1),jb,kb,true), &u(1,ib,jb,kb,true),
            &dev_SideEdge_Recv[g][idx_up], &dev_SideEdge_Recv[g][idx_down], ni, nj, nk, m_ppadding,
            m_neighbor[0],  m_neighbor[1], MPI_PROC_NULL );
      CheckCudaCall(hipGetLastError(), "HaloToBufferKernel<<<,>>>(...)", __FILE__, __LINE__);

    }
}
//-----------------------------------------------------------------------

void EW::unpack_HaloArrayCU_Y( Sarray& u, int g , int st)
{
  REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
	    << " nc = " << u.m_nc );
  int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
  MPI_Status status;
  dim3 gridsize, blocksize;
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize.x  = 1 * 1 * m_gpu_gridsize[2];
  gridsize.y  = 1;
  gridsize.z  = 1;
  blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize.y = 1;
  blocksize.z = 1;

  int ni = m_iEnd[g] - m_iStart[g] + 1;
  int nj = m_jEnd[g] - m_jStart[g] + 1;
  int nk = m_kEnd[g] - m_kStart[g] + 1;
  int n_m_ppadding1 = 3*nj*nk*m_ppadding;
  int n_m_ppadding2 = 3*ni*nk*m_ppadding;
  int idx_left = 0;
  int idx_right = n_m_ppadding2;
  int idx_up = 2*n_m_ppadding2;
  int idx_down = 2*n_m_ppadding2 + n_m_ppadding1;

  if( u.m_nc == 3 )
    {
      // Y-direction communication
      if(m_corder)
        hipLaunchKernelGGL(HaloToBufferKernelX_dev_rev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st] ,  &u(1,ib,jb,kb,true), &u(1,ib,je-(m_ppadding-1),kb,true),
            &dev_SideEdge_Recv[g][idx_left], &dev_SideEdge_Recv[g][idx_right],
            ni, nj, nk, m_ppadding, m_neighbor[2], m_neighbor[3], MPI_PROC_NULL );
      else
        hipLaunchKernelGGL(HaloToBufferKernelX_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  &u(1,ib,jb,kb,true), &u(1,ib,je-(m_ppadding-1),kb,true),
            &dev_SideEdge_Recv[g][idx_left], &dev_SideEdge_Recv[g][idx_right],
            ni, nj, nk, m_ppadding, m_neighbor[2], m_neighbor[3], MPI_PROC_NULL );
      CheckCudaCall(hipGetLastError(), "HaloToBufferKernel<<<,>>>(...)", __FILE__, __LINE__);
      
    }
}

//-----------------------------------------------------------------------

void EW::communicate_arrayCU_X( Sarray& u, int g , int st)
{
   REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
             << " nc = " << u.m_nc );
   int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
   MPI_Status status;
   hipError_t retcode;
   dim3 gridsize, blocksize;
   //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.x  = 1 * 1 * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;

   int ni = m_iEnd[g] - m_iStart[g] + 1;
   int nj = m_jEnd[g] - m_jStart[g] + 1;
   int nk = m_kEnd[g] - m_kStart[g] + 1;
   int n_m_ppadding1 = 3*nj*nk*m_ppadding;
   int n_m_ppadding2 = 3*ni*nk*m_ppadding;
   int idx_left = 0;
   int idx_right = n_m_ppadding2;
   int idx_up = 2*n_m_ppadding2;
   int idx_down = 2*n_m_ppadding2 + n_m_ppadding1;

   if( u.m_nc == 1 )
   {
      int xtag1 = 345;
      int xtag2 = 346;
      int ytag1 = 347;
      int ytag2 = 348;
      int grid = g;
      // X-direction communication
      MPI_Sendrecv( &u(ie-(2*m_ppadding-1),jb,kb,true), 1, m_send_type1[2*grid], m_neighbor[1], xtag1,
                    &u(ib,jb,kb,true), 1, m_send_type1[2*grid], m_neighbor[0], xtag1,
                    m_cartesian_communicator, &status );
      MPI_Sendrecv( &u(ib+m_ppadding,jb,kb,true), 1, m_send_type1[2*grid], m_neighbor[0], xtag2,
                    &u(ie-(m_ppadding-1),jb,kb,true), 1, m_send_type1[2*grid], m_neighbor[1], xtag2,
                    m_cartesian_communicator, &status );
   }
   else if( u.m_nc == 3 )
   {
      int xtag1 = 345;
      int xtag2 = 346;
      int ytag1 = 347;
      int ytag2 = 348;

      // Packing / unpacking of the array to send into the linear memory communication buffers 
      // is done outside of this subroutine.

#ifdef SW4_CUDA_AWARE_MPI

      SafeCudaCall( hipStreamSynchronize(m_cuobj->m_stream[st]) );

#ifdef SW4_NONBLOCKING
      // X-direction communication with non-blocking MPI
      // First post receives, then send edges to neighbors
      // TODO: Waits on Isends may be done lazily for more optimization,
      //       only the recvs are needed to proceed
      MPI_Request requests[4];
      MPI_Status statuses[4];
      MPI_Irecv(&dev_SideEdge_Recv[g][idx_down], n_m_ppadding1, m_mpifloat, m_neighbor[0],
                xtag1, m_cartesian_communicator, &requests[0]);
      MPI_Irecv(&dev_SideEdge_Recv[g][idx_up], n_m_ppadding1, m_mpifloat, m_neighbor[1],
                xtag2, m_cartesian_communicator, &requests[1]);
      MPI_Isend(&dev_SideEdge_Send[g][idx_up], n_m_ppadding1, m_mpifloat, m_neighbor[1],
                xtag1, m_cartesian_communicator, &requests[2]);
      MPI_Isend(&dev_SideEdge_Send[g][idx_down], n_m_ppadding1, m_mpifloat, m_neighbor[0],
                xtag2, m_cartesian_communicator, &requests[3]);
      MPI_Waitall(4, requests, statuses);
#else
      // X-direction communication
      MPI_Sendrecv(&dev_SideEdge_Send[g][idx_up], n_m_ppadding1, m_mpifloat,
                   m_neighbor[1], xtag1, &dev_SideEdge_Recv[g][idx_down],
		   n_m_ppadding1, m_mpifloat, m_neighbor[0], xtag1, m_cartesian_communicator, &status);

      MPI_Sendrecv(&dev_SideEdge_Send[g][idx_down], n_m_ppadding1, m_mpifloat,
                   m_neighbor[0], xtag2, &dev_SideEdge_Recv[g][idx_up],
		   n_m_ppadding1, m_mpifloat, m_neighbor[1], xtag2, m_cartesian_communicator, &status);
#endif // SW4_NONBLOCKING

#else

      // Copy buffers to host
      if (m_neighbor[1] != MPI_PROC_NULL)
        hipMemcpyAsync(&m_SideEdge_Send[g][idx_up], &dev_SideEdge_Send[g][idx_up],
                        n_m_ppadding1*sizeof(float_sw4), hipMemcpyDeviceToHost, m_cuobj->m_stream[st]);

      if (m_neighbor[0] != MPI_PROC_NULL)
        hipMemcpyAsync(&m_SideEdge_Send[g][idx_down], &dev_SideEdge_Send[g][idx_down],
                        n_m_ppadding1*sizeof(float_sw4), hipMemcpyDeviceToHost, m_cuobj->m_stream[st]);
      retcode = hipStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != hipSuccess )
        {
          cout << "Error communicate_array hipMemcpy returned (DeviceToHost) "
               << hipGetErrorString(retcode) << endl;
          exit(1);
        }

#ifdef SW4_NONBLOCKING
      // X-direction communication with non-blocking MPI
      // First post receives, then send edges to neighbors
      // TODO: Waits on Isends may be done lazily for more optimization,
      //       only the recvs are needed to proceed
      MPI_Request requests[4];
      MPI_Status statuses[4];
      MPI_Irecv(&m_SideEdge_Recv[g][idx_down], n_m_ppadding1, m_mpifloat, m_neighbor[0],
                xtag1, m_cartesian_communicator, &requests[0]);
      MPI_Irecv(&m_SideEdge_Recv[g][idx_up], n_m_ppadding1, m_mpifloat, m_neighbor[1],
                xtag2, m_cartesian_communicator, &requests[1]);
      MPI_Isend(&m_SideEdge_Send[g][idx_up], n_m_ppadding1, m_mpifloat, m_neighbor[1],
                xtag1, m_cartesian_communicator, &requests[2]);
      MPI_Isend(&m_SideEdge_Send[g][idx_down], n_m_ppadding1, m_mpifloat, m_neighbor[0],
                xtag2, m_cartesian_communicator, &requests[3]);
      MPI_Waitall(4, requests, statuses);
#else
      // Send and receive with MPI
      MPI_Sendrecv(&m_SideEdge_Send[g][idx_up], n_m_ppadding1, m_mpifloat,
                   m_neighbor[1], xtag1, &m_SideEdge_Recv[g][idx_down],
                   n_m_ppadding1, m_mpifloat, m_neighbor[0], xtag1, m_cartesian_communicator, &status);

      MPI_Sendrecv(&m_SideEdge_Send[g][idx_down], n_m_ppadding1, m_mpifloat,
                   m_neighbor[0], xtag2, &m_SideEdge_Recv[g][idx_up],
                   n_m_ppadding1, m_mpifloat, m_neighbor[1], xtag2, m_cartesian_communicator, &status);
#endif // SW4_NONBLOCKING

      // Copy buffers to device
      if (m_neighbor[1] != MPI_PROC_NULL)
        hipMemcpyAsync(&dev_SideEdge_Recv[g][idx_up], &m_SideEdge_Recv[g][idx_up],
                        n_m_ppadding1*sizeof(float_sw4), hipMemcpyHostToDevice, m_cuobj->m_stream[st] );

      if (m_neighbor[0] != MPI_PROC_NULL)
        hipMemcpyAsync(&dev_SideEdge_Recv[g][idx_down], &m_SideEdge_Recv[g][idx_down],
                        n_m_ppadding1*sizeof(float_sw4), hipMemcpyHostToDevice, m_cuobj->m_stream[st] );

      retcode = hipStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != hipSuccess )
        {
          cout << "Error communicate_array hipMemcpy returned (Host2Device) "
               << hipGetErrorString(retcode) << endl;
          exit(1);
        }

#endif
   }
}
//-----------------------------------------------------------------------

void EW::communicate_arrayCU_Y( Sarray& u, int g , int st)
{
   REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
             << " nc = " << u.m_nc );
   int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
   MPI_Status status;
   hipError_t retcode;
   dim3 gridsize, blocksize;
   //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.x  = 1 * 1 * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;

   int ni = m_iEnd[g] - m_iStart[g] + 1;
   int nj = m_jEnd[g] - m_jStart[g] + 1;
   int nk = m_kEnd[g] - m_kStart[g] + 1;
   int n_m_ppadding1 = 3*nj*nk*m_ppadding;
   int n_m_ppadding2 = 3*ni*nk*m_ppadding;
   int idx_left = 0;
   int idx_right = n_m_ppadding2;
   int idx_up = 2*n_m_ppadding2;
   int idx_down = 2*n_m_ppadding2 + n_m_ppadding1;

   if( u.m_nc == 1 )
   {
      int xtag1 = 345;
      int xtag2 = 346;
      int ytag1 = 347;
      int ytag2 = 348;
      int grid = g;
      //Y-direction communication
      MPI_Sendrecv( &u(ib,je-(2*m_ppadding-1),kb,true), 1, m_send_type1[2*grid+1], m_neighbor[3], ytag1,
                    &u(ib,jb,kb,true), 1, m_send_type1[2*grid+1], m_neighbor[2], ytag1,
                    m_cartesian_communicator, &status );
      MPI_Sendrecv( &u(ib,jb+m_ppadding,kb,true), 1, m_send_type1[2*grid+1], m_neighbor[2], ytag2,
                    &u(ib,je-(m_ppadding-1),kb,true), 1, m_send_type1[2*grid+1], m_neighbor[3], ytag2,
                    m_cartesian_communicator, &status );
   }
   else if( u.m_nc == 3 )
   {
      int xtag1 = 345;
      int xtag2 = 346;
      int ytag1 = 347;
      int ytag2 = 348;

      // Packing / unpacking of the array to send into the linear memory communication buffers 
      // is done outside of this subroutine.

#ifdef SW4_CUDA_AWARE_MPI

      SafeCudaCall( hipStreamSynchronize(m_cuobj->m_stream[st]) );

#ifdef SW4_NONBLOCKING
      // Y-direction communication with non-blocking MPI
      // First post receives, then send edges to neighbors
      // TODO: Waits on Isends may be done lazily for more optimization,
      //       only the recvs are needed to proceed
      MPI_Request requests[4];
      MPI_Status statuses[4];
      MPI_Irecv(&dev_SideEdge_Recv[g][idx_right], n_m_ppadding2, m_mpifloat, m_neighbor[3],
                ytag1, m_cartesian_communicator, &requests[0]);
      MPI_Irecv(&dev_SideEdge_Recv[g][idx_left], n_m_ppadding2, m_mpifloat, m_neighbor[2],
                ytag2, m_cartesian_communicator, &requests[1]);
      MPI_Isend(&dev_SideEdge_Send[g][idx_left], n_m_ppadding2, m_mpifloat, m_neighbor[2],
                ytag1, m_cartesian_communicator, &requests[2]);
      MPI_Isend(&dev_SideEdge_Send[g][idx_right], n_m_ppadding2, m_mpifloat, m_neighbor[3],
                ytag2, m_cartesian_communicator, &requests[3]);
      MPI_Waitall(4, requests, statuses);
#else
      // Y-direction communication
      MPI_Sendrecv(&dev_SideEdge_Send[g][idx_right], n_m_ppadding2, m_mpifloat,
                   m_neighbor[3], ytag2, &dev_SideEdge_Recv[g][idx_left],
                   n_m_ppadding2, m_mpifloat, m_neighbor[2], ytag2, m_cartesian_communicator, &status);
      MPI_Sendrecv(&dev_SideEdge_Send[g][idx_left], n_m_ppadding2, m_mpifloat,
                   m_neighbor[2], ytag1, &dev_SideEdge_Recv[g][idx_right],
                   n_m_ppadding2, m_mpifloat, m_neighbor[3], ytag1, m_cartesian_communicator, &status);
#endif // SW4_NONBLOCKING

#else

      // Copy buffers to host
      if (m_neighbor[2] != MPI_PROC_NULL)
        hipMemcpyAsync(&m_SideEdge_Send[g][idx_left], &dev_SideEdge_Send[g][idx_left],
                        n_m_ppadding2*sizeof(float_sw4), hipMemcpyDeviceToHost, m_cuobj->m_stream[st]);

      if (m_neighbor[3] != MPI_PROC_NULL)
        hipMemcpyAsync(&m_SideEdge_Send[g][idx_right], &dev_SideEdge_Send[g][idx_right],
                        n_m_ppadding2*sizeof(float_sw4), hipMemcpyDeviceToHost, m_cuobj->m_stream[st]);
      
      retcode = hipStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != hipSuccess )
        {
          cout << "Error communicate_array hipMemcpy returned (DeviceToHost) "
               << hipGetErrorString(retcode) << endl;
          exit(1);
        }

#ifdef SW4_NONBLOCKING
      // Y-direction communication with non-blocking MPI
      // First post receives, then send edges to neighbors
      // TODO: Waits on Isends may be done lazily for more optimization,
      //       only the recvs are needed to proceed
      MPI_Request requests[4];
      MPI_Status statuses[4];
      MPI_Irecv(&m_SideEdge_Recv[g][idx_right], n_m_ppadding2, m_mpifloat, m_neighbor[3],
                ytag1, m_cartesian_communicator, &requests[0]);
      MPI_Irecv(&m_SideEdge_Recv[g][idx_left], n_m_ppadding2, m_mpifloat, m_neighbor[2],
                ytag2, m_cartesian_communicator, &requests[1]);
      MPI_Isend(&m_SideEdge_Send[g][idx_left], n_m_ppadding2, m_mpifloat, m_neighbor[2],
                ytag1, m_cartesian_communicator, &requests[2]);
      MPI_Isend(&m_SideEdge_Send[g][idx_right], n_m_ppadding2, m_mpifloat, m_neighbor[3],
                ytag2, m_cartesian_communicator, &requests[3]);
      MPI_Waitall(4, requests, statuses);
#else
      // Send and receive with MPI
      MPI_Sendrecv(&m_SideEdge_Send[g][idx_left], n_m_ppadding2, m_mpifloat,
                   m_neighbor[2], ytag1, &m_SideEdge_Recv[g][idx_right],
                   n_m_ppadding2, m_mpifloat, m_neighbor[3], ytag1, m_cartesian_communicator, &status);

      MPI_Sendrecv(&m_SideEdge_Send[g][idx_right], n_m_ppadding2, m_mpifloat,
                   m_neighbor[3], ytag2, &m_SideEdge_Recv[g][idx_left],
                   n_m_ppadding2, m_mpifloat, m_neighbor[2], ytag2, m_cartesian_communicator, &status);      
#endif // SW4_NONBLOCKING

      // Copy buffers to device
      if (m_neighbor[2] != MPI_PROC_NULL)
        hipMemcpyAsync(&dev_SideEdge_Recv[g][idx_left], &m_SideEdge_Recv[g][idx_left],
                        n_m_ppadding2*sizeof(float_sw4), hipMemcpyHostToDevice, m_cuobj->m_stream[st] );

      if (m_neighbor[3] != MPI_PROC_NULL)
        hipMemcpyAsync(&dev_SideEdge_Recv[g][idx_right], &m_SideEdge_Recv[g][idx_right],
                        n_m_ppadding2*sizeof(float_sw4), hipMemcpyHostToDevice, m_cuobj->m_stream[st] );
      
      retcode = hipStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != hipSuccess )
        {
          cout << "Error communicate_array hipMemcpy returned (Host2Device) "
               << hipGetErrorString(retcode) << endl;
          exit(1);
        }

#endif
   }
}

//-----------------------------------------------------------------------

void EW::setup_device_communication_array()
{
  dev_SideEdge_Send.resize(mNumberOfGrids);
  dev_SideEdge_Recv.resize(mNumberOfGrids);
#ifndef SW4_CUDA_AWARE_MPI
  m_SideEdge_Send.resize(mNumberOfGrids);
  m_SideEdge_Recv.resize(mNumberOfGrids);
#endif

  if( m_ndevice > 0 )
  {
     hipError_t retcode;

     for( int g=0 ; g<mNumberOfGrids; g++)
     {

        int ni = m_iEnd[g] - m_iStart[g] + 1;
        int nj = m_jEnd[g] - m_jStart[g] + 1;
        int nk = m_kEnd[g] - m_kStart[g] + 1;
        int n_m_ppadding1 = 3*nj*nk*m_ppadding;
        int n_m_ppadding2 = 3*ni*nk*m_ppadding;

        retcode = hipMalloc( (void**)&dev_SideEdge_Send[g], sizeof(float_sw4)*2*(n_m_ppadding1+n_m_ppadding2) );
        if( retcode != hipSuccess )
           cout << "Error, EW::setup_device_communication_arra hipMalloc returned "
                << hipGetErrorString(retcode) << endl;

        retcode = hipMalloc( (void**)&dev_SideEdge_Recv[g], sizeof(float_sw4)*2*(n_m_ppadding1+n_m_ppadding2) );
        if( retcode != hipSuccess )
           cout << "Error, EW::setup_device_communication_arra hipMalloc returned "
                << hipGetErrorString(retcode) << endl;

#ifndef SW4_CUDA_AWARE_MPI

        retcode = hipHostMalloc( (void**)&m_SideEdge_Send[g], sizeof(float_sw4)*2*(n_m_ppadding1+n_m_ppadding2) );
        if( retcode != hipSuccess )
           cout << "Error, EW::setup_device_communication_arra hipHostMalloc returned "
                << hipGetErrorString(retcode) << endl;

        retcode = hipHostMalloc( (void**)&m_SideEdge_Recv[g], sizeof(float_sw4)*2*(n_m_ppadding1+n_m_ppadding2) );
        if( retcode != hipSuccess )
           cout << "Error, EW::setup_device_communication_arra hipHostMalloc returned "
                << hipGetErrorString(retcode) << endl;

#endif

     }
  }
}

//-----------------------------------------------------------------------
void EW::enforceBCCU( vector<Sarray> & a_U, vector<Sarray>& a_Mu, vector<Sarray>& a_Lambda,
                      float_sw4 t, vector<float_sw4**> & a_BCForcing, int st )
{
  float_sw4 om=0, ph=0, cv=0;
  for(int g=0 ; g<mNumberOfGrids; g++ )
    bcfortsg_gpu( m_iStart[g], m_iEnd[g], m_jStart[g], m_jEnd[g], m_kStart[g], m_kEnd[g],
                  dev_BndryWindow[g], m_global_nx[g], m_global_ny[g], m_global_nz[g], a_U[g].dev_ptr(),
                  mGridSize[g], dev_bcType[g], a_Mu[g].dev_ptr(), a_Lambda[g].dev_ptr(),
                  t, dev_BCForcing[g][0], dev_BCForcing[g][1], dev_BCForcing[g][2],
                  dev_BCForcing[g][3], dev_BCForcing[g][4], dev_BCForcing[g][5],
                  om, ph, cv, dev_sg_str_x[g], dev_sg_str_y[g], m_corder, m_cuobj->m_stream[st]);
}

//-----------------------------------------------------------------------
void EW::allocateTimeSeriesOnDeviceCU( int& nvals, int& ntloc, int*& i0dev,
				       int*& j0dev, int*& k0dev, int*& g0dev,
				       int*& modedev, float_sw4**& urec_dev,
				       float_sw4**& urec_host, float_sw4**& urec_hdev )
{
   // urec_dev:  array of pointers on device pointing to device memory
   // urec_host: array of pointers on host pointing to host memory
   // urec_hdev: array of pointers on host pointing to device memory
  vector<int> i0vect, j0vect, k0vect, g0vect;
  vector<int> modevect;
  // Save location and type of stations, and count their number (ntloc), 
  // and the total size of memory needed (nvals)
  for (int ts=0; ts<m_GlobalTimeSeries.size(); ts++)
  {
     if ( m_GlobalTimeSeries[ts]->myPoint())
     {
	i0vect.push_back(m_GlobalTimeSeries[ts]->m_i0);
	j0vect.push_back(m_GlobalTimeSeries[ts]->m_j0);
	k0vect.push_back(m_GlobalTimeSeries[ts]->m_k0);
	g0vect.push_back(m_GlobalTimeSeries[ts]->m_grid0);
	modevect.push_back(m_GlobalTimeSeries[ts]->getMode());
	nvals += m_GlobalTimeSeries[ts]->urec_size();
     }
  }
  ntloc=i0vect.size();

  // Allocate memory on host
  urec_host = new float_sw4*[ntloc];
  urec_hdev = new float_sw4*[ntloc]; 
  hipError_t  retval;
  if( ntloc > 0 )
  {
  // Allocate memory on device, and copy the location to vectors on device 
     retval = hipMalloc( (void**)&i0dev, sizeof(int)*ntloc);
     if( retval != hipSuccess )
	cout << "Error in hipMalloc of i0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMalloc( (void**)&j0dev, sizeof(int)*ntloc);
     if( retval != hipSuccess )
	cout << "Error in hipMalloc of j0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMalloc( (void**)&k0dev, sizeof(int)*ntloc);
     if( retval != hipSuccess )
	cout << "Error in hipMalloc of k0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMalloc( (void**)&g0dev, sizeof(int)*ntloc);
     if( retval != hipSuccess )
	cout << "Error in hipMalloc of g0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMalloc( (void**)&modedev, sizeof(int)*ntloc);
     if( retval != hipSuccess )
	cout << "Error in hipMalloc of modedev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMemcpy( i0dev,  &i0vect[0], sizeof(int)*ntloc, hipMemcpyHostToDevice );
     if( retval != hipSuccess )
	cout << "Error in cudaMmemcpy of i0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMemcpy( j0dev,  &j0vect[0], sizeof(int)*ntloc, hipMemcpyHostToDevice );
     if( retval != hipSuccess )
	cout << "Error in cudaMmemcpy of j0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMemcpy( k0dev,  &k0vect[0], sizeof(int)*ntloc, hipMemcpyHostToDevice );
     if( retval != hipSuccess )
	cout << "Error in cudaMmemcpy of k0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMemcpy( g0dev,  &g0vect[0], sizeof(int)*ntloc, hipMemcpyHostToDevice );
     if( retval != hipSuccess )
	cout << "Error in cudaMmemcpy of g0dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMemcpy( modedev,  &modevect[0], sizeof(int)*ntloc, hipMemcpyHostToDevice );
     if( retval != hipSuccess )
	cout << "Error in cudaMmemcpy of modedev retval = " <<hipGetErrorString(retval) << endl;

    // Allocate memory on host and and device to hold the data 
     float_sw4* devmem;
     retval = hipMalloc( (void**)&devmem, sizeof(float_sw4)*nvals);
     float_sw4* hostmem = new float_sw4[nvals];

     size_t ptr=0;
     int tsnr=0;
     for( int ts=0; ts<m_GlobalTimeSeries.size(); ts++)
     {
	if ( m_GlobalTimeSeries[ts]->myPoint() )
	{
	   urec_hdev[tsnr] = &devmem[ptr];
	   urec_host[tsnr] = &hostmem[ptr];
	   ptr += m_GlobalTimeSeries[ts]->urec_size();
	   tsnr++;
	}
     }
    // Create and allocate pointer to pointers on device
     retval = hipMalloc( (void**)&urec_dev, sizeof(float_sw4*)*ntloc);
     if( retval != hipSuccess )
	cout << "Error in hipMalloc of urec_dev retval = " <<hipGetErrorString(retval) << endl;
     retval = hipMemcpy( urec_dev, urec_hdev, sizeof(float_sw4*)*ntloc, hipMemcpyHostToDevice);
     if( retval != hipSuccess )
	cout << "Error in cudaMmemcpy of urec_dev retval = " <<hipGetErrorString(retval) << endl;
  }
}

//-----------------------------------------------------------------------
void EW::extractRecordDataCU( int nt, int* mode, int* i0v, int* j0v, int* k0v,
			      int* g0v, float_sw4** urec_dev, Sarray* dev_Um, Sarray* dev_U,
			      float_sw4 dt, float_sw4* h_dev, Sarray* dev_metric,
			      Sarray* dev_j, int st, int nvals, float_sw4* urec_hostmem,
			      float_sw4* urec_devmem )
{
   dim3 blocksize, gridsize;
   gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize.y  = 1;
   gridsize.z  = 1;
   blocksize.x = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize.y = 1;
   blocksize.z = 1;
   hipLaunchKernelGGL(extractRecordData_dev, dim3(gridsize), dim3(blocksize), 0, m_cuobj->m_stream[st],  
		          nt, mode, i0v, j0v,
		          k0v, g0v, urec_dev, dev_Um, dev_U,
			  dt, h_dev, mNumberOfCartesianGrids, dev_metric, dev_j );
   hipError_t retval = hipMemcpy( urec_hostmem, urec_devmem, sizeof(float_sw4)*nvals, hipMemcpyDeviceToHost );
   if( retval != hipSuccess )
      cout << "Error in hipMemcpy in EW::extractRecordDataCU retval = " <<hipGetErrorString(retval) << endl;
   
}

