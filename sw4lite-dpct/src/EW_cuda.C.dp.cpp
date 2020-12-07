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
#include "EW.h"

#include "EWCuda.h"

#include "device-routines.h"

  
//-----------------------------------------------------------------------
void EW::evalRHSCU(vector<Sarray> & a_U, vector<Sarray>& a_Mu, vector<Sarray>& a_Lambda,
		   vector<Sarray> & a_Uacc, int st )
{
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0];
   gridsize[1] = m_gpu_gridsize[1];
   gridsize[0] = m_gpu_gridsize[2];
   blocksize[2] = m_gpu_blocksize[0];
   blocksize[1] = m_gpu_blocksize[1];
   blocksize[0] = m_gpu_blocksize[2];
   for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
   {
      if( m_corder ) 
      {
        int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
        int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
        int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
        sycl::range<3> block(1, RHS4_BLOCKY, RHS4_BLOCKX);
        sycl::range<3> grid(1, 1, 1);
        grid[2] = (ni + block[2] - 1) / block[2];
        grid[1] = (nj + block[1] - 1) / block[1];
        grid[0] = 1;
        /*
DPCT1049:126: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            sycl::range<3> a_uSh1_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);
            sycl::range<3> a_uSh2_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);
            sycl::range<3> a_uSh3_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);

            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh1_acc_ct1(a_uSh1_range_ct1, cgh);
            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh2_acc_ct1(a_uSh2_range_ct1, cgh);
            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh3_acc_ct1(a_uSh3_range_ct1, cgh);

            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_Uacc_g_dev_ptr_ct6 = a_Uacc[g].dev_ptr();
            auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
            auto a_Mu_g_dev_ptr_ct8 = a_Mu[g].dev_ptr();
            auto a_Lambda_g_dev_ptr_ct9 = a_Lambda[g].dev_ptr();
            auto mGridSize_g_ct10 = mGridSize[g];
            auto dev_sg_str_x_g_ct11 = dev_sg_str_x[g];
            auto dev_sg_str_y_g_ct12 = dev_sg_str_y[g];
            auto dev_sg_str_z_g_ct13 = dev_sg_str_z[g];
            auto m_ghost_points_ct14 = m_ghost_points;

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                rhs4center_dev_rev_v2(
                                    m_iStart_g_ct0, m_iEnd_g_ct1,
                                    m_jStart_g_ct2, m_jEnd_g_ct3,
                                    m_kStart_g_ct4, m_kEnd_g_ct5,
                                    a_Uacc_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                    a_Mu_g_dev_ptr_ct8, a_Lambda_g_dev_ptr_ct9,
                                    mGridSize_g_ct10, dev_sg_str_x_g_ct11,
                                    dev_sg_str_y_g_ct12, dev_sg_str_z_g_ct13,
                                    m_ghost_points_ct14, item_ct1,
                                    dpct::accessor<float_sw4, dpct::local, 3>(
                                        a_uSh1_acc_ct1, a_uSh1_range_ct1),
                                    dpct::accessor<float_sw4, dpct::local, 3>(
                                        a_uSh2_acc_ct1, a_uSh2_range_ct1),
                                    dpct::accessor<float_sw4, dpct::local, 3>(
                                        a_uSh3_acc_ct1, a_uSh3_range_ct1));
                             });
         });
      }
      else
      {
        int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
        int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
        int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
        sycl::range<3> block(1, RHS4_BLOCKY, RHS4_BLOCKX);
        sycl::range<3> grid(1, 1, 1);
        grid[2] = (ni + block[2] - 1) / block[2];
        grid[1] = (nj + block[1] - 1) / block[1];
        grid[0] = 1;
        /*
DPCT1049:127: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            sycl::range<3> a_uSh1_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);
            sycl::range<3> a_uSh2_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);
            sycl::range<3> a_uSh3_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);

            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh1_acc_ct1(a_uSh1_range_ct1, cgh);
            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh2_acc_ct1(a_uSh2_range_ct1, cgh);
            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh3_acc_ct1(a_uSh3_range_ct1, cgh);

            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_Uacc_g_dev_ptr_ct6 = a_Uacc[g].dev_ptr();
            auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
            auto a_Mu_g_dev_ptr_ct8 = a_Mu[g].dev_ptr();
            auto a_Lambda_g_dev_ptr_ct9 = a_Lambda[g].dev_ptr();
            auto mGridSize_g_ct10 = mGridSize[g];
            auto dev_sg_str_x_g_ct11 = dev_sg_str_x[g];
            auto dev_sg_str_y_g_ct12 = dev_sg_str_y[g];
            auto dev_sg_str_z_g_ct13 = dev_sg_str_z[g];
            auto m_ghost_points_ct14 = m_ghost_points;

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                rhs4center_dev_v2(
                                    m_iStart_g_ct0, m_iEnd_g_ct1,
                                    m_jStart_g_ct2, m_jEnd_g_ct3,
                                    m_kStart_g_ct4, m_kEnd_g_ct5,
                                    a_Uacc_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                    a_Mu_g_dev_ptr_ct8, a_Lambda_g_dev_ptr_ct9,
                                    mGridSize_g_ct10, dev_sg_str_x_g_ct11,
                                    dev_sg_str_y_g_ct12, dev_sg_str_z_g_ct13,
                                    m_ghost_points_ct14, item_ct1,
                                    dpct::accessor<float_sw4, dpct::local, 3>(
                                        a_uSh1_acc_ct1, a_uSh1_range_ct1),
                                    dpct::accessor<float_sw4, dpct::local, 3>(
                                        a_uSh2_acc_ct1, a_uSh2_range_ct1),
                                    dpct::accessor<float_sw4, dpct::local, 3>(
                                        a_uSh3_acc_ct1, a_uSh3_range_ct1));
                             });
         });
      }
   }
// Boundary operator at upper boundary
   blocksize[0] = 1;
   gridsize[0] = 6;

   for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
   {
      if( m_onesided[g][4] )
      {
	 if( m_corder )
            /*
DPCT1049:128: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               extern dpct::constant_memory<float_sw4, 1> dev_acof;
               extern dpct::constant_memory<float_sw4, 1> dev_ghcof;
               extern dpct::constant_memory<float_sw4, 1> dev_bope;

               dev_acof.init(*m_cuobj->m_stream[st]);
               dev_ghcof.init(*m_cuobj->m_stream[st]);
               dev_bope.init(*m_cuobj->m_stream[st]);

               auto dev_acof_ptr_ct1 = dev_acof.get_ptr();
               auto dev_ghcof_ptr_ct1 = dev_ghcof.get_ptr();
               auto dev_bope_ptr_ct1 = dev_bope.get_ptr();

               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Uacc_g_dev_ptr_ct6 = a_Uacc[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Mu_g_dev_ptr_ct8 = a_Mu[g].dev_ptr();
               auto a_Lambda_g_dev_ptr_ct9 = a_Lambda[g].dev_ptr();
               auto mGridSize_g_ct10 = mGridSize[g];
               auto dev_sg_str_x_g_ct11 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct12 = dev_sg_str_y[g];
               auto dev_sg_str_z_g_ct13 = dev_sg_str_z[g];
               auto m_ghost_points_ct14 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      rhs4upper_dev_rev(
                          m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                          m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                          a_Uacc_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                          a_Mu_g_dev_ptr_ct8, a_Lambda_g_dev_ptr_ct9,
                          mGridSize_g_ct10, dev_sg_str_x_g_ct11,
                          dev_sg_str_y_g_ct12, dev_sg_str_z_g_ct13,
                          m_ghost_points_ct14, item_ct1, dev_acof_ptr_ct1,
                          dev_ghcof_ptr_ct1, dev_bope_ptr_ct1);
                   });
            });
         else
            /*
DPCT1049:129: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               extern dpct::constant_memory<float_sw4, 1> dev_acof;
               extern dpct::constant_memory<float_sw4, 1> dev_ghcof;
               extern dpct::constant_memory<float_sw4, 1> dev_bope;

               dev_acof.init(*m_cuobj->m_stream[st]);
               dev_ghcof.init(*m_cuobj->m_stream[st]);
               dev_bope.init(*m_cuobj->m_stream[st]);

               auto dev_acof_ptr_ct1 = dev_acof.get_ptr();
               auto dev_ghcof_ptr_ct1 = dev_ghcof.get_ptr();
               auto dev_bope_ptr_ct1 = dev_bope.get_ptr();

               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Uacc_g_dev_ptr_ct6 = a_Uacc[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Mu_g_dev_ptr_ct8 = a_Mu[g].dev_ptr();
               auto a_Lambda_g_dev_ptr_ct9 = a_Lambda[g].dev_ptr();
               auto mGridSize_g_ct10 = mGridSize[g];
               auto dev_sg_str_x_g_ct11 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct12 = dev_sg_str_y[g];
               auto dev_sg_str_z_g_ct13 = dev_sg_str_z[g];
               auto m_ghost_points_ct14 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      rhs4upper_dev(
                          m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                          m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                          a_Uacc_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                          a_Mu_g_dev_ptr_ct8, a_Lambda_g_dev_ptr_ct9,
                          mGridSize_g_ct10, dev_sg_str_x_g_ct11,
                          dev_sg_str_y_g_ct12, dev_sg_str_z_g_ct13,
                          m_ghost_points_ct14, item_ct1, dev_acof_ptr_ct1,
                          dev_ghcof_ptr_ct1, dev_bope_ptr_ct1);
                   });
            });
      }
   }

   if( m_topography_exists )
   {
      gridsize[2] = m_gpu_gridsize[0];
      gridsize[1] = m_gpu_gridsize[1];
      gridsize[0] = m_gpu_gridsize[2];
      blocksize[2] = m_gpu_blocksize[0];
      blocksize[1] = m_gpu_blocksize[1];
      blocksize[0] = m_gpu_blocksize[2];
      int g=mNumberOfGrids-1;
   // Boundary operator at upper boundary
      int onesided4 = 0;  
      if( m_onesided[g][4] )
      {
        onesided4 = 1;
        blocksize[0] = 1;
        gridsize[0] = 6;
        if( m_corder )
          /*
DPCT1049:130: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               extern dpct::constant_memory<float_sw4, 1> dev_acof;
               extern dpct::constant_memory<float_sw4, 1> dev_ghcof;
               extern dpct::constant_memory<float_sw4, 1> dev_bope;

               dev_acof.init(*m_cuobj->m_stream[st]);
               dev_ghcof.init(*m_cuobj->m_stream[st]);
               dev_bope.init(*m_cuobj->m_stream[st]);

               auto dev_acof_ptr_ct1 = dev_acof.get_ptr();
               auto dev_ghcof_ptr_ct1 = dev_ghcof.get_ptr();
               auto dev_bope_ptr_ct1 = dev_bope.get_ptr();

               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_U_g_dev_ptr_ct6 = a_U[g].dev_ptr();
               auto a_Mu_g_dev_ptr_ct7 = a_Mu[g].dev_ptr();
               auto a_Lambda_g_dev_ptr_ct8 = a_Lambda[g].dev_ptr();
               auto mMetric_dev_ptr_ct9 = mMetric.dev_ptr();
               auto mJ_dev_ptr_ct10 = mJ.dev_ptr();
               auto a_Uacc_g_dev_ptr_ct11 = a_Uacc[g].dev_ptr();
               auto dev_sg_str_x_g_ct12 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct13 = dev_sg_str_y[g];
               auto m_ghost_points_ct14 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      rhs4sgcurvupper_dev_rev(
                          m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                          m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                          a_U_g_dev_ptr_ct6, a_Mu_g_dev_ptr_ct7,
                          a_Lambda_g_dev_ptr_ct8, mMetric_dev_ptr_ct9,
                          mJ_dev_ptr_ct10, a_Uacc_g_dev_ptr_ct11,
                          dev_sg_str_x_g_ct12, dev_sg_str_y_g_ct13,
                          m_ghost_points_ct14, item_ct1, dev_acof_ptr_ct1,
                          dev_ghcof_ptr_ct1, dev_bope_ptr_ct1);
                   });
            });
        else
          /*
DPCT1049:131: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               extern dpct::constant_memory<float_sw4, 1> dev_acof;
               extern dpct::constant_memory<float_sw4, 1> dev_ghcof;
               extern dpct::constant_memory<float_sw4, 1> dev_bope;

               dev_acof.init(*m_cuobj->m_stream[st]);
               dev_ghcof.init(*m_cuobj->m_stream[st]);
               dev_bope.init(*m_cuobj->m_stream[st]);

               auto dev_acof_ptr_ct1 = dev_acof.get_ptr();
               auto dev_ghcof_ptr_ct1 = dev_ghcof.get_ptr();
               auto dev_bope_ptr_ct1 = dev_bope.get_ptr();

               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_U_g_dev_ptr_ct6 = a_U[g].dev_ptr();
               auto a_Mu_g_dev_ptr_ct7 = a_Mu[g].dev_ptr();
               auto a_Lambda_g_dev_ptr_ct8 = a_Lambda[g].dev_ptr();
               auto mMetric_dev_ptr_ct9 = mMetric.dev_ptr();
               auto mJ_dev_ptr_ct10 = mJ.dev_ptr();
               auto a_Uacc_g_dev_ptr_ct11 = a_Uacc[g].dev_ptr();
               auto dev_sg_str_x_g_ct12 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct13 = dev_sg_str_y[g];
               auto m_ghost_points_ct14 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      rhs4sgcurvupper_dev(
                          m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                          m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                          a_U_g_dev_ptr_ct6, a_Mu_g_dev_ptr_ct7,
                          a_Lambda_g_dev_ptr_ct8, mMetric_dev_ptr_ct9,
                          mJ_dev_ptr_ct10, a_Uacc_g_dev_ptr_ct11,
                          dev_sg_str_x_g_ct12, dev_sg_str_y_g_ct13,
                          m_ghost_points_ct14, item_ct1, dev_acof_ptr_ct1,
                          dev_ghcof_ptr_ct1, dev_bope_ptr_ct1);
                   });
            });
      }
      gridsize[0] = m_gpu_gridsize[2];
      blocksize[0] = m_gpu_blocksize[2];
      if( m_corder )
      {
        int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
        int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
        int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
        sycl::range<3> block(1, RHS4_BLOCKY, RHS4_BLOCKX);
        sycl::range<3> grid(1, 1, 1);
        grid[2] = (ni + block[2] - 1) / block[2];
        grid[1] = (nj + block[1] - 1) / block[1];
        grid[0] = 1;
        /*
DPCT1049:133: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            sycl::range<3> a_uSh1_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);
            sycl::range<3> a_uSh2_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);
            sycl::range<3> a_uSh3_range_ct1(5 /*DIAMETER*/,
                                            20 /*RHS4_BLOCKY+2*RADIUS*/,
                                            20 /*RHS4_BLOCKX+2*RADIUS*/);

            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh1_acc_ct1(a_uSh1_range_ct1, cgh);
            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh2_acc_ct1(a_uSh2_range_ct1, cgh);
            sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                           sycl::access::target::local>
                a_uSh3_acc_ct1(a_uSh3_range_ct1, cgh);

            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_U_g_dev_ptr_ct6 = a_U[g].dev_ptr();
            auto a_Mu_g_dev_ptr_ct7 = a_Mu[g].dev_ptr();
            auto a_Lambda_g_dev_ptr_ct8 = a_Lambda[g].dev_ptr();
            auto mMetric_dev_ptr_ct9 = mMetric.dev_ptr();
            auto mJ_dev_ptr_ct10 = mJ.dev_ptr();
            auto a_Uacc_g_dev_ptr_ct11 = a_Uacc[g].dev_ptr();
            auto dev_sg_str_x_g_ct13 = dev_sg_str_x[g];
            auto dev_sg_str_y_g_ct14 = dev_sg_str_y[g];
            auto m_ghost_points_ct15 = m_ghost_points;

            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item_ct1) {
                   rhs4sgcurv_dev_rev_v2(
                       m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                       m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                       a_U_g_dev_ptr_ct6, a_Mu_g_dev_ptr_ct7,
                       a_Lambda_g_dev_ptr_ct8, mMetric_dev_ptr_ct9,
                       mJ_dev_ptr_ct10, a_Uacc_g_dev_ptr_ct11, onesided4,
                       dev_sg_str_x_g_ct13, dev_sg_str_y_g_ct14,
                       m_ghost_points_ct15, item_ct1,
                       dpct::accessor<float_sw4, dpct::local, 3>(
                           a_uSh1_acc_ct1, a_uSh1_range_ct1),
                       dpct::accessor<float_sw4, dpct::local, 3>(
                           a_uSh2_acc_ct1, a_uSh2_range_ct1),
                       dpct::accessor<float_sw4, dpct::local, 3>(
                           a_uSh3_acc_ct1, a_uSh3_range_ct1));
                });
         });
      }
      else
        /*
DPCT1049:132: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_U_g_dev_ptr_ct6 = a_U[g].dev_ptr();
            auto a_Mu_g_dev_ptr_ct7 = a_Mu[g].dev_ptr();
            auto a_Lambda_g_dev_ptr_ct8 = a_Lambda[g].dev_ptr();
            auto mMetric_dev_ptr_ct9 = mMetric.dev_ptr();
            auto mJ_dev_ptr_ct10 = mJ.dev_ptr();
            auto a_Uacc_g_dev_ptr_ct11 = a_Uacc[g].dev_ptr();
            auto dev_sg_str_x_g_ct13 = dev_sg_str_x[g];
            auto dev_sg_str_y_g_ct14 = dev_sg_str_y[g];
            auto m_ghost_points_ct15 = m_ghost_points;

            cgh.parallel_for(
                sycl::nd_range<3>(gridsize * blocksize, blocksize),
                [=](sycl::nd_item<3> item_ct1) {
                   rhs4sgcurv_dev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                                  m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                                  a_U_g_dev_ptr_ct6, a_Mu_g_dev_ptr_ct7,
                                  a_Lambda_g_dev_ptr_ct8, mMetric_dev_ptr_ct9,
                                  mJ_dev_ptr_ct10, a_Uacc_g_dev_ptr_ct11,
                                  onesided4, dev_sg_str_x_g_ct13,
                                  dev_sg_str_y_g_ct14, m_ghost_points_ct15,
                                  item_ct1);
                });
         });
   }
}

//-----------------------------------------------------------------------
void EW::evalPredictorCU( vector<Sarray> & a_Up, vector<Sarray> & a_U, vector<Sarray> & a_Um,
			  vector<Sarray>& a_Rho, vector<Sarray> & a_Lu, vector<Sarray> & a_F, int st )
{
   float_sw4 dt2 = mDt*mDt;
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;
   for( int g=0 ; g<mNumberOfGrids; g++ )
   {
      if( m_corder )
         /*
DPCT1049:134: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
            auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
            auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
            auto a_Lu_g_dev_ptr_ct9 = a_Lu[g].dev_ptr();
            auto a_F_g_dev_ptr_ct10 = a_F[g].dev_ptr();
            auto a_Rho_g_dev_ptr_ct11 = a_Rho[g].dev_ptr();
            auto m_ghost_points_ct13 = m_ghost_points;

            cgh.parallel_for(
                sycl::nd_range<3>(gridsize * blocksize, blocksize),
                [=](sycl::nd_item<3> item_ct1) {
                   pred_dev_rev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                                m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                                a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                a_Um_g_dev_ptr_ct8, a_Lu_g_dev_ptr_ct9,
                                a_F_g_dev_ptr_ct10, a_Rho_g_dev_ptr_ct11, dt2,
                                m_ghost_points_ct13, item_ct1);
                });
         });
      else
         /*
DPCT1049:135: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
            auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
            auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
            auto a_Lu_g_dev_ptr_ct9 = a_Lu[g].dev_ptr();
            auto a_F_g_dev_ptr_ct10 = a_F[g].dev_ptr();
            auto a_Rho_g_dev_ptr_ct11 = a_Rho[g].dev_ptr();
            auto m_ghost_points_ct13 = m_ghost_points;

            cgh.parallel_for(
                sycl::nd_range<3>(gridsize * blocksize, blocksize),
                [=](sycl::nd_item<3> item_ct1) {
                   pred_dev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                            m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                            a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                            a_Um_g_dev_ptr_ct8, a_Lu_g_dev_ptr_ct9,
                            a_F_g_dev_ptr_ct10, a_Rho_g_dev_ptr_ct11, dt2,
                            m_ghost_points_ct13, item_ct1);
                });
         });
   }
}

//---------------------------------------------------------------------------
void EW::evalCorrectorCU( vector<Sarray> & a_Up, vector<Sarray>& a_Rho,
			  vector<Sarray> & a_Lu, vector<Sarray> & a_F, int st )
{
   float_sw4 dt4 = mDt*mDt*mDt*mDt;
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;
   for( int g=0 ; g<mNumberOfGrids; g++ )
   {
      if( m_corder )
         /*
DPCT1049:136: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
            auto a_Lu_g_dev_ptr_ct7 = a_Lu[g].dev_ptr();
            auto a_F_g_dev_ptr_ct8 = a_F[g].dev_ptr();
            auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
            auto m_ghost_points_ct11 = m_ghost_points;

            cgh.parallel_for(
                sycl::nd_range<3>(gridsize * blocksize, blocksize),
                [=](sycl::nd_item<3> item_ct1) {
                   corr_dev_rev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                                m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                                a_Up_g_dev_ptr_ct6, a_Lu_g_dev_ptr_ct7,
                                a_F_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9, dt4,
                                m_ghost_points_ct11, item_ct1);
                });
         });
      else
         /*
DPCT1049:137: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto m_iStart_g_ct0 = m_iStart[g];
            auto m_iEnd_g_ct1 = m_iEnd[g];
            auto m_jStart_g_ct2 = m_jStart[g];
            auto m_jEnd_g_ct3 = m_jEnd[g];
            auto m_kStart_g_ct4 = m_kStart[g];
            auto m_kEnd_g_ct5 = m_kEnd[g];
            auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
            auto a_Lu_g_dev_ptr_ct7 = a_Lu[g].dev_ptr();
            auto a_F_g_dev_ptr_ct8 = a_F[g].dev_ptr();
            auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
            auto m_ghost_points_ct11 = m_ghost_points;

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                corr_dev(m_iStart_g_ct0, m_iEnd_g_ct1,
                                         m_jStart_g_ct2, m_jEnd_g_ct3,
                                         m_kStart_g_ct4, m_kEnd_g_ct5,
                                         a_Up_g_dev_ptr_ct6, a_Lu_g_dev_ptr_ct7,
                                         a_F_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                                         dt4, m_ghost_points_ct11, item_ct1);
                             });
         });
   }
}

//---------------------------------------------------------------------------
void EW::evalDpDmInTimeCU(vector<Sarray> & a_Up, vector<Sarray> & a_U, vector<Sarray> & a_Um,
			  vector<Sarray> & a_Uacc, int st )
{
   float_sw4 dt2i = 1./(mDt*mDt);
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;
   for(int g=0 ; g<mNumberOfGrids; g++ )
   {
      /*
DPCT1049:138: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
      m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
         auto m_iStart_g_ct0 = m_iStart[g];
         auto m_iEnd_g_ct1 = m_iEnd[g];
         auto m_jStart_g_ct2 = m_jStart[g];
         auto m_jEnd_g_ct3 = m_jEnd[g];
         auto m_kStart_g_ct4 = m_kStart[g];
         auto m_kEnd_g_ct5 = m_kEnd[g];
         auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
         auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
         auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
         auto a_Uacc_g_dev_ptr_ct9 = a_Uacc[g].dev_ptr();
         auto m_ghost_points_ct11 = m_ghost_points;

         cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                          [=](sycl::nd_item<3> item_ct1) {
                             dpdmt_dev(m_iStart_g_ct0, m_iEnd_g_ct1,
                                       m_jStart_g_ct2, m_jEnd_g_ct3,
                                       m_kStart_g_ct4, m_kEnd_g_ct5,
                                       a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                       a_Um_g_dev_ptr_ct8, a_Uacc_g_dev_ptr_ct9,
                                       dt2i, m_ghost_points_ct11, item_ct1);
                          });
      });
   }
}

//-----------------------------------------------------------------------
void EW::addSuperGridDampingCU(vector<Sarray> & a_Up, vector<Sarray> & a_U,
			     vector<Sarray> & a_Um, vector<Sarray> & a_Rho, int st )
{
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0];
   gridsize[1] = m_gpu_gridsize[1];
   gridsize[0] = m_gpu_gridsize[2];
   blocksize[2] = m_gpu_blocksize[0];
   blocksize[1] = m_gpu_blocksize[1];
   blocksize[0] = m_gpu_blocksize[2];
   for(int g=0 ; g<mNumberOfCartesianGrids; g++ )
   {
      if( m_sg_damping_order == 4 )
      {
	 if( m_corder )
         {
           int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
           int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
           int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
           const sycl::range<3> block(1, ADDSGD4_BLOCKY, ADDSGD4_BLOCKX);
           sycl::range<3> grid(1, 1, 1);
           grid[2] = (ni + block[2] - 1) / block[2];
           grid[1] = (nj + block[1] - 1) / block[1];
           grid[0] = 1;
           /*
DPCT1049:139: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               sycl::range<3> su_range_ct1(3, 28 /*ADDSGD4_BLOCKX+2*RADIUS*/,
                                           20 /*ADDSGD4_BLOCKY+2*RADIUS*/);
               sycl::range<3> sum_range_ct1(3, 28 /*ADDSGD4_BLOCKX+2*RADIUS*/,
                                            20 /*ADDSGD4_BLOCKY+2*RADIUS*/);

               sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                              sycl::access::target::local>
                   su_acc_ct1(su_range_ct1, cgh);
               sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                              sycl::access::target::local>
                   sum_acc_ct1(sum_range_ct1, cgh);

               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_dc_z_g_ct12 = dev_sg_dc_z[g];
               auto dev_sg_str_x_g_ct13 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct14 = dev_sg_str_y[g];
               auto dev_sg_str_z_g_ct15 = dev_sg_str_z[g];
               auto dev_sg_corner_x_g_ct16 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct17 = dev_sg_corner_y[g];
               auto dev_sg_corner_z_g_ct18 = dev_sg_corner_z[g];
               auto m_supergrid_damping_coefficient_ct19 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct20 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(grid * block, block),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd4_dev_rev_v2(
                          m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                          m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                          a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                          a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                          dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                          dev_sg_dc_z_g_ct12, dev_sg_str_x_g_ct13,
                          dev_sg_str_y_g_ct14, dev_sg_str_z_g_ct15,
                          dev_sg_corner_x_g_ct16, dev_sg_corner_y_g_ct17,
                          dev_sg_corner_z_g_ct18,
                          m_supergrid_damping_coefficient_ct19,
                          m_ghost_points_ct20, item_ct1,
                          dpct::accessor<float_sw4, dpct::local, 3>(
                              su_acc_ct1, su_range_ct1),
                          dpct::accessor<float_sw4, dpct::local, 3>(
                              sum_acc_ct1, sum_range_ct1));
                   });
            });
         }
	 else
         {
           int ni = m_iEnd[g] - m_iStart[g] + 1 - 2*m_ghost_points;
           int nj = m_jEnd[g] - m_jStart[g] + 1 - 2*m_ghost_points;
           int nk = m_kEnd[g] - m_kStart[g] + 1 - 2*m_ghost_points;
           const sycl::range<3> block(1, ADDSGD4_BLOCKY, ADDSGD4_BLOCKX);
           sycl::range<3> grid(1, 1, 1);
           grid[2] = (ni + block[2] - 1) / block[2];
           grid[1] = (nj + block[1] - 1) / block[1];
           grid[0] = 1;
           /*
DPCT1049:140: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               sycl::range<3> su_range_ct1(28 /*ADDSGD4_BLOCKX+2*RADIUS*/,
                                           20 /*ADDSGD4_BLOCKY+2*RADIUS*/, 3);
               sycl::range<3> sum_range_ct1(28 /*ADDSGD4_BLOCKX+2*RADIUS*/,
                                            20 /*ADDSGD4_BLOCKY+2*RADIUS*/, 3);

               sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                              sycl::access::target::local>
                   su_acc_ct1(su_range_ct1, cgh);
               sycl::accessor<float_sw4, 3, sycl::access::mode::read_write,
                              sycl::access::target::local>
                   sum_acc_ct1(sum_range_ct1, cgh);

               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_dc_z_g_ct12 = dev_sg_dc_z[g];
               auto dev_sg_str_x_g_ct13 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct14 = dev_sg_str_y[g];
               auto dev_sg_str_z_g_ct15 = dev_sg_str_z[g];
               auto dev_sg_corner_x_g_ct16 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct17 = dev_sg_corner_y[g];
               auto dev_sg_corner_z_g_ct18 = dev_sg_corner_z[g];
               auto m_supergrid_damping_coefficient_ct19 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct20 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(grid * block, block),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd4_dev_v2(
                          m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                          m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                          a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                          a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                          dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                          dev_sg_dc_z_g_ct12, dev_sg_str_x_g_ct13,
                          dev_sg_str_y_g_ct14, dev_sg_str_z_g_ct15,
                          dev_sg_corner_x_g_ct16, dev_sg_corner_y_g_ct17,
                          dev_sg_corner_z_g_ct18,
                          m_supergrid_damping_coefficient_ct19,
                          m_ghost_points_ct20, item_ct1,
                          dpct::accessor<float_sw4, dpct::local, 3>(
                              su_acc_ct1, su_range_ct1),
                          dpct::accessor<float_sw4, dpct::local, 3>(
                              sum_acc_ct1, sum_range_ct1));
                   });
            });
          }
      }
      else if(  m_sg_damping_order == 6 )
      {
	 if( m_corder )
            /*
DPCT1049:141: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_dc_z_g_ct12 = dev_sg_dc_z[g];
               auto dev_sg_str_x_g_ct13 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct14 = dev_sg_str_y[g];
               auto dev_sg_str_z_g_ct15 = dev_sg_str_z[g];
               auto dev_sg_corner_x_g_ct16 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct17 = dev_sg_corner_y[g];
               auto dev_sg_corner_z_g_ct18 = dev_sg_corner_z[g];
               auto m_supergrid_damping_coefficient_ct19 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct20 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd6_dev_rev(
                          m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                          m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                          a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                          a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                          dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                          dev_sg_dc_z_g_ct12, dev_sg_str_x_g_ct13,
                          dev_sg_str_y_g_ct14, dev_sg_str_z_g_ct15,
                          dev_sg_corner_x_g_ct16, dev_sg_corner_y_g_ct17,
                          dev_sg_corner_z_g_ct18,
                          m_supergrid_damping_coefficient_ct19,
                          m_ghost_points_ct20, item_ct1);
                   });
            });
         else
            /*
DPCT1049:142: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_dc_z_g_ct12 = dev_sg_dc_z[g];
               auto dev_sg_str_x_g_ct13 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct14 = dev_sg_str_y[g];
               auto dev_sg_str_z_g_ct15 = dev_sg_str_z[g];
               auto dev_sg_corner_x_g_ct16 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct17 = dev_sg_corner_y[g];
               auto dev_sg_corner_z_g_ct18 = dev_sg_corner_z[g];
               auto m_supergrid_damping_coefficient_ct19 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct20 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd6_dev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                                  m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                                  a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                  a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                                  dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                                  dev_sg_dc_z_g_ct12, dev_sg_str_x_g_ct13,
                                  dev_sg_str_y_g_ct14, dev_sg_str_z_g_ct15,
                                  dev_sg_corner_x_g_ct16,
                                  dev_sg_corner_y_g_ct17,
                                  dev_sg_corner_z_g_ct18,
                                  m_supergrid_damping_coefficient_ct19,
                                  m_ghost_points_ct20, item_ct1);
                   });
            });
      }
   }

   if( m_topography_exists )
   {
      int g=mNumberOfGrids-1;
      if( m_sg_damping_order == 4 )
      {
	 if( m_corder )
            /*
DPCT1049:143: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_str_x_g_ct12 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct13 = dev_sg_str_y[g];
               auto mJ_dev_ptr_ct14 = mJ.dev_ptr();
               auto dev_sg_corner_x_g_ct15 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct16 = dev_sg_corner_y[g];
               auto m_supergrid_damping_coefficient_ct17 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct18 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd4c_dev_rev(m_iStart_g_ct0, m_iEnd_g_ct1,
                                       m_jStart_g_ct2, m_jEnd_g_ct3,
                                       m_kStart_g_ct4, m_kEnd_g_ct5,
                                       a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                       a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                                       dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                                       dev_sg_str_x_g_ct12, dev_sg_str_y_g_ct13,
                                       mJ_dev_ptr_ct14, dev_sg_corner_x_g_ct15,
                                       dev_sg_corner_y_g_ct16,
                                       m_supergrid_damping_coefficient_ct17,
                                       m_ghost_points_ct18, item_ct1);
                   });
            });
         else
            /*
DPCT1049:144: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_str_x_g_ct12 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct13 = dev_sg_str_y[g];
               auto mJ_dev_ptr_ct14 = mJ.dev_ptr();
               auto dev_sg_corner_x_g_ct15 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct16 = dev_sg_corner_y[g];
               auto m_supergrid_damping_coefficient_ct17 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct18 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd4c_dev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                                   m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                                   a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                   a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                                   dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                                   dev_sg_str_x_g_ct12, dev_sg_str_y_g_ct13,
                                   mJ_dev_ptr_ct14, dev_sg_corner_x_g_ct15,
                                   dev_sg_corner_y_g_ct16,
                                   m_supergrid_damping_coefficient_ct17,
                                   m_ghost_points_ct18, item_ct1);
                   });
            });
      }
      else if(  m_sg_damping_order == 6 )
      {
	 if( m_corder )
            /*
DPCT1049:145: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_str_x_g_ct12 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct13 = dev_sg_str_y[g];
               auto mJ_dev_ptr_ct14 = mJ.dev_ptr();
               auto dev_sg_corner_x_g_ct15 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct16 = dev_sg_corner_y[g];
               auto m_supergrid_damping_coefficient_ct17 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct18 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd6c_dev_rev(m_iStart_g_ct0, m_iEnd_g_ct1,
                                       m_jStart_g_ct2, m_jEnd_g_ct3,
                                       m_kStart_g_ct4, m_kEnd_g_ct5,
                                       a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                       a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                                       dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                                       dev_sg_str_x_g_ct12, dev_sg_str_y_g_ct13,
                                       mJ_dev_ptr_ct14, dev_sg_corner_x_g_ct15,
                                       dev_sg_corner_y_g_ct16,
                                       m_supergrid_damping_coefficient_ct17,
                                       m_ghost_points_ct18, item_ct1);
                   });
            });
         else
            /*
DPCT1049:146: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
            m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
               auto m_iStart_g_ct0 = m_iStart[g];
               auto m_iEnd_g_ct1 = m_iEnd[g];
               auto m_jStart_g_ct2 = m_jStart[g];
               auto m_jEnd_g_ct3 = m_jEnd[g];
               auto m_kStart_g_ct4 = m_kStart[g];
               auto m_kEnd_g_ct5 = m_kEnd[g];
               auto a_Up_g_dev_ptr_ct6 = a_Up[g].dev_ptr();
               auto a_U_g_dev_ptr_ct7 = a_U[g].dev_ptr();
               auto a_Um_g_dev_ptr_ct8 = a_Um[g].dev_ptr();
               auto a_Rho_g_dev_ptr_ct9 = a_Rho[g].dev_ptr();
               auto dev_sg_dc_x_g_ct10 = dev_sg_dc_x[g];
               auto dev_sg_dc_y_g_ct11 = dev_sg_dc_y[g];
               auto dev_sg_str_x_g_ct12 = dev_sg_str_x[g];
               auto dev_sg_str_y_g_ct13 = dev_sg_str_y[g];
               auto mJ_dev_ptr_ct14 = mJ.dev_ptr();
               auto dev_sg_corner_x_g_ct15 = dev_sg_corner_x[g];
               auto dev_sg_corner_y_g_ct16 = dev_sg_corner_y[g];
               auto m_supergrid_damping_coefficient_ct17 =
                   m_supergrid_damping_coefficient;
               auto m_ghost_points_ct18 = m_ghost_points;

               cgh.parallel_for(
                   sycl::nd_range<3>(gridsize * blocksize, blocksize),
                   [=](sycl::nd_item<3> item_ct1) {
                      addsgd6c_dev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                                   m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                                   a_Up_g_dev_ptr_ct6, a_U_g_dev_ptr_ct7,
                                   a_Um_g_dev_ptr_ct8, a_Rho_g_dev_ptr_ct9,
                                   dev_sg_dc_x_g_ct10, dev_sg_dc_y_g_ct11,
                                   dev_sg_str_x_g_ct12, dev_sg_str_y_g_ct13,
                                   mJ_dev_ptr_ct14, dev_sg_corner_x_g_ct15,
                                   dev_sg_corner_y_g_ct16,
                                   m_supergrid_damping_coefficient_ct17,
                                   m_ghost_points_ct18, item_ct1);
                   });
            });
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
void EW::copy_supergrid_arrays_to_device() try {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
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
     int retcode;
     for( int g=0 ; g<mNumberOfGrids; g++) 
     {
	// sg_str
        /*
DPCT1003:183: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_str_x[g] = sycl::malloc_device<float>(
                 (m_iEnd[g] - m_iStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:148: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:147: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc x "
                  "returned "
               /*
               DPCT1009:184: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:185: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_str_y[g] = sycl::malloc_device<float>(
                 (m_jEnd[g] - m_jStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:150: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:149: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc y "
                  "returned "
               /*
               DPCT1009:186: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:187: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_str_z[g] = sycl::malloc_device<float>(
                 (m_kEnd[g] - m_kStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:152: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:151: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc z "
                  "returned "
               /*
               DPCT1009:188: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:189: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_str_x[g], m_sg_str_x[g],
                         sizeof(float_sw4) * (m_iEnd[g] - m_iStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:154: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:153: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy x "
                  "returned "
               /*
               DPCT1009:190: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:191: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_str_y[g], m_sg_str_y[g],
                         sizeof(float_sw4) * (m_jEnd[g] - m_jStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:156: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:155: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy y "
                  "returned "
               /*
               DPCT1009:192: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:193: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_str_z[g], m_sg_str_z[g],
                         sizeof(float_sw4) * (m_kEnd[g] - m_kStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:158: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:157: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy z "
                  "returned "
               /*
               DPCT1009:194: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;

        // sg_dc
        /*
DPCT1003:195: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_dc_x[g] = sycl::malloc_device<float>(
                 (m_iEnd[g] - m_iStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:160: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:159: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc dc_x "
                  "returned "
               /*
               DPCT1009:196: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:197: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_dc_y[g] = sycl::malloc_device<float>(
                 (m_jEnd[g] - m_jStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:162: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:161: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc dc_y "
                  "returned "
               /*
               DPCT1009:198: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:199: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_dc_z[g] = sycl::malloc_device<float>(
                 (m_kEnd[g] - m_kStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:164: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:163: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc dc_z "
                  "returned "
               /*
               DPCT1009:200: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:201: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_dc_x[g], m_sg_dc_x[g],
                         sizeof(float_sw4) * (m_iEnd[g] - m_iStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:166: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:165: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy dc_x "
                  "returned "
               /*
               DPCT1009:202: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:203: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_dc_y[g], m_sg_dc_y[g],
                         sizeof(float_sw4) * (m_jEnd[g] - m_jStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:168: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:167: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy dc_y "
                  "returned "
               /*
               DPCT1009:204: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:205: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_dc_z[g], m_sg_dc_z[g],
                         sizeof(float_sw4) * (m_kEnd[g] - m_kStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:170: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:169: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy dc_z "
                  "returned "
               /*
               DPCT1009:206: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        // sg_corner
        /*
DPCT1003:207: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_corner_x[g] = sycl::malloc_device<float>(
                 (m_iEnd[g] - m_iStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:172: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:171: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc "
                  "corner_x returned "
               /*
               DPCT1009:208: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:209: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_corner_y[g] = sycl::malloc_device<float>(
                 (m_jEnd[g] - m_jStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:174: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:173: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc "
                  "corner_y returned "
               /*
               DPCT1009:210: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:211: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dev_sg_corner_z[g] = sycl::malloc_device<float>(
                 (m_kEnd[g] - m_kStart[g] + 1), dpct::get_default_queue()),
             0);
        /*
DPCT1000:176: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:175: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMalloc "
                  "corner_z returned "
               /*
               DPCT1009:212: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:213: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_corner_x[g], m_sg_corner_x[g],
                         sizeof(float_sw4) * (m_iEnd[g] - m_iStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:178: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:177: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy "
                  "corner_x returned "
               /*
               DPCT1009:214: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:215: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_corner_y[g], m_sg_corner_y[g],
                         sizeof(float_sw4) * (m_jEnd[g] - m_jStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:180: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:179: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy "
                  "corner_y returned "
               /*
               DPCT1009:216: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
        /*
DPCT1003:217: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode =
            (dpct::get_default_queue()
                 .memcpy(dev_sg_corner_z[g], m_sg_corner_z[g],
                         sizeof(float_sw4) * (m_kEnd[g] - m_kStart[g] + 1))
                 .wait(),
             0);
        /*
DPCT1000:182: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:181: The statement could not be removed.
*/
           cout
               << "Error, EW::copy_supergrid_arrays_to_device cudaMemcpy "
                  "corner_z returned "
               /*
               DPCT1009:218: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;
     }
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
void EW::find_cuda_device() try {
   int retcode;
   dpct::device_info prop;
   /*
DPCT1003:221: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
   retcode = (m_ndevice = dpct::dev_mgr::instance().device_count(), 0);
   /*
DPCT1000:220: Error handling if-stmt was detected but could not be rewritten.
*/
   if (retcode != 0)
   {
      /*
DPCT1001:219: The statement could not be removed.
*/
      cout << "Error from cudaGetDeviceCount: Error string = " <<
          /*
          DPCT1009:222: SYCL uses exceptions to report errors and does not use
          the error codes. The original code was commented out and a warning
          string was inserted. You need to rewrite this code.
          */
          "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
           << endl;
   }
   // Note: This will not work if some nodes have GPU and others do not
   // It is assumed that all nodes are identical wrt GPU
   if( m_ndevice > 0 && m_myrank == 0 )
   {
      cout << m_ndevice << " Cuda devices found:" << endl;
      for( int i=0 ;  i < m_ndevice ; i++ )
      {
         /*
DPCT1003:223: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
         retcode = (dpct::dev_mgr::instance().get_device(i).get_device_info(prop), 0);
         cout << "      Device " << i << ": name = " << prop.get_name() <<
             /*
             DPCT1005:224: The device version is different. You need to rewrite
             this code.
             */
             ",  Compute capability:"
              << prop.get_major_version() << "." << prop.get_minor_version()
              << ",  Memory (GB) " << (prop.get_global_mem_size() >> 30)
              << endl;
      }
   }
   //Added following line for all ranks
   /*
DPCT1003:225: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
   /*
   retcode = (dpct::dev_mgr::instance().get_device(0).get_device_info(prop), 0);

   // Check block size
   CHECK_INPUT( m_gpu_blocksize[0] <= prop.maxThreadsDim[0],
		"Error: max block x " << m_gpu_blocksize[0] << " too large\n");
   CHECK_INPUT( m_gpu_blocksize[1] <= prop.maxThreadsDim[1], 
		"Error: max block y " << m_gpu_blocksize[1] << " too large\n");
   CHECK_INPUT( m_gpu_blocksize[2] <= prop.maxThreadsDim[2],
		"Error: max block z " << m_gpu_blocksize[2] << " too large\n");
   CHECK_INPUT(m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2] <=
                   prop.get_max_work_group_size(),
               "Error: max number of threads per block "
                   << prop.get_max_work_group_size() << " is exceeded \n");
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
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//-----------------------------------------------------------------------
bool EW::check_for_nan_GPU( vector<Sarray>& a_U, int verbose, string name )
{
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;

   int  *retval_dev, retval_host = 0;
   int  *idx_dev, idx_host = 0;
   int cnan, inan, jnan, knan;
   int  nijk, nij, ni;

   retval_dev = sycl::malloc_device<int>(1, dpct::get_default_queue());
   dpct::get_default_queue().memcpy(retval_dev, &retval_host, sizeof(int)).wait();
   idx_dev = sycl::malloc_device<int>(1, dpct::get_default_queue());
   dpct::get_default_queue().memcpy(idx_dev, &idx_host, sizeof(int)).wait();

   for( int g=0 ; g<mNumberOfGrids; g++ )
   {
      /*
DPCT1049:226: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
         auto m_iStart_g_ct0 = m_iStart[g];
         auto m_iEnd_g_ct1 = m_iEnd[g];
         auto m_jStart_g_ct2 = m_jStart[g];
         auto m_jEnd_g_ct3 = m_jEnd[g];
         auto m_kStart_g_ct4 = m_kStart[g];
         auto m_kEnd_g_ct5 = m_kEnd[g];
         auto a_U_g_dev_ptr_ct6 = a_U[g].dev_ptr();

         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                check_nan_dev(m_iStart_g_ct0, m_iEnd_g_ct1, m_jStart_g_ct2,
                              m_jEnd_g_ct3, m_kStart_g_ct4, m_kEnd_g_ct5,
                              a_U_g_dev_ptr_ct6, retval_dev, idx_dev, item_ct1);
             });
      });

      dpct::get_default_queue().memcpy(&retval_host, retval_dev, sizeof(int)).wait();

      if ( retval_host != 0) 
      {
         dpct::get_default_queue().memcpy(&idx_host, idx_dev, sizeof(int)).wait();

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
   sycl::range<3> blocksize(1, 1, 1), gridsize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;
   /*
DPCT1049:227: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
   m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
      auto mNumberOfGrids_ct2 = mNumberOfGrids;
      auto dev_point_sources_ct3 = dev_point_sources;
      auto m_point_sources_size_ct4 = m_point_sources.size();
      auto dev_identsources_ct5 = dev_identsources;
      auto m_identsources_size_ct6 = m_identsources.size();

      cgh.parallel_for(
          sycl::nd_range<3>(gridsize * blocksize, blocksize),
          [=](sycl::nd_item<3> item_ct1) {
             forcing_dev(t, dev_F, mNumberOfGrids_ct2, dev_point_sources_ct3,
                         m_point_sources_size_ct4, dev_identsources_ct5,
                         m_identsources_size_ct6, tt, item_ct1);
          });
   });
}

//-----------------------------------------------------------------------
void EW::init_point_sourcesCU( )
{
   sycl::range<3> blocksize(1, 1, 1), gridsize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;
   /*
DPCT1049:228: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
   dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto dev_point_sources_ct0 = dev_point_sources;
      auto m_point_sources_size_ct1 = m_point_sources.size();

      cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                       [=](sycl::nd_item<3> item_ct1) {
                          init_forcing_dev(dev_point_sources_ct0,
                                           m_point_sources_size_ct1, item_ct1);
                       });
   });
}

void EW::cartesian_bc_forcingCU(float_sw4 t, vector<float_sw4 **> &a_BCForcing,
                                vector<Source *> &a_sources, int st)
    // assign the boundary forcing arrays dev_BCForcing[g][side]
    try {
  int retcode;
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
          /*
DPCT1003:231: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
          retcode = (m_cuobj->m_stream[st]->memcpy(
                         dev_BCForcing[g][side], a_BCForcing[g][side], nBytes),
                     0);
          /*
DPCT1000:230: Error handling if-stmt was detected but could not be rewritten.
*/
          if (retcode != 0)
            /*
DPCT1001:229: The statement could not be removed.
*/
            /*
DPCT1009:232: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
            cout
                << "Error, EW::cartesian_bc_forcing_CU cudaMemcpy x returned "
                << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
                << endl;
        }
        else
        {
          m_cuobj->m_stream[st]->memset(dev_BCForcing[g][side], 0, nBytes);
        }
      }
    }
    else
    {
      for( int side=0 ; side < 6 ; side++ )
      {
        size_t nBytes = sizeof(float_sw4)*3*m_NumberOfBCPoints[g][side];
        m_cuobj->m_stream[st]->memset(dev_BCForcing[g][side], 0, nBytes);
      }
    }
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//-----------------------------------------------------------------------

void EW::copy_bcforcing_arrays_to_device() try {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();

   //Set up boundary data array on the deivec
  if(m_ndevice > 0 )
  {
    int retcode;
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
          /*
DPCT1003:237: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
          retcode = (dev_BCForcing[g][side] = (float *)sycl::malloc_device(
                         nBytes, dpct::get_default_queue()),
                     0);
          /*
DPCT1000:234: Error handling if-stmt was detected but could not be rewritten.
*/
          if (retcode != 0)
          {
             /*
DPCT1001:233: The statement could not be removed.
*/
             /*
DPCT1009:238: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
             cout
                 << "Error, EW::copy_bcforcing_arrays_to_device cudaMalloc x "
                    "returned "
                 << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
                 << endl;
             exit(-1);
          }
          /*
DPCT1003:239: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
          retcode =
              (dpct::get_default_queue()
                   .memcpy(dev_BCForcing[g][side], BCForcing[g][side], nBytes)
                   .wait(),
               0);
          /*
DPCT1000:236: Error handling if-stmt was detected but could not be rewritten.
*/
          if (retcode != 0)
          {
            /*
DPCT1001:235: The statement could not be removed.
*/
            /*
DPCT1009:240: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
            cout
                << "Error, EW::copy_bcforcing_arrays_to_device cudaMemcpy x "
                   "returned "
                << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
                << endl;
            exit(-1);
          }
        }
      }
    }
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//-----------------------------------------------------------------------

void EW::copy_bctype_arrays_to_device() try {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
  // Set up boundary type array on the deivec
  if(m_ndevice > 0 )
  {
    int retcode;
    dev_bcType.resize(mNumberOfGrids);
    for( int g = 0; g <mNumberOfGrids; g++ )
    {
      size_t nBytes = sizeof(boundaryConditionType)*6;
      /*
DPCT1003:245: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
      retcode = (dev_bcType[g] = (boundaryConditionType *)sycl::malloc_device(
                     nBytes, dpct::get_default_queue()),
                 0);
      /*
DPCT1000:242: Error handling if-stmt was detected but could not be rewritten.
*/
      if (retcode != 0)
        /*
DPCT1001:241: The statement could not be removed.
*/
        /*
DPCT1009:246: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error, EW::copy_bctype_arrays_to_device cudaMalloc x returned "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
            << endl;
      /*
DPCT1003:247: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
      retcode = (dpct::get_default_queue()
                     .memcpy(dev_bcType[g], m_bcType[g], nBytes)
                     .wait(),
                 0);
      /*
DPCT1000:244: Error handling if-stmt was detected but could not be rewritten.
*/
      if (retcode != 0)
        /*
DPCT1001:243: The statement could not be removed.
*/
        /*
DPCT1009:248: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error, EW::copy_bctype_arrays_to_device cudaMemcpy x returned "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
            << endl;
    }
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//-----------------------------------------------------------------------

void EW::copy_bndrywindow_arrays_to_device() try {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
  //Set up boundary window array on the deivec
  if(m_ndevice > 0 )
  {
    int retcode;
    dev_BndryWindow.resize(mNumberOfGrids);
    for( int g = 0; g <mNumberOfGrids; g++ )
    {
      size_t nBytes = sizeof(int)*36;
      /*
DPCT1003:253: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
      retcode = (dev_BndryWindow[g] = (int *)sycl::malloc_device(
                     nBytes, dpct::get_default_queue()),
                 0);
      /*
DPCT1000:250: Error handling if-stmt was detected but could not be rewritten.
*/
      if (retcode != 0)
        /*
DPCT1001:249: The statement could not be removed.
*/
        /*
DPCT1009:254: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error, EW::copy_bndrywindow_arrays_to_device cudaMalloc x "
               "returned "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
            << endl;
      /*
DPCT1003:255: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
      retcode = (dpct::get_default_queue()
                     .memcpy(dev_BndryWindow[g], m_BndryWindow[g], nBytes)
                     .wait(),
                 0);
      /*
DPCT1000:252: Error handling if-stmt was detected but could not be rewritten.
*/
      if (retcode != 0)
        /*
DPCT1001:251: The statement could not be removed.
*/
        /*
DPCT1009:256: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error, EW::copy_bndrywindow_arrays_to_device cudaMemcpy x "
               "returned "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
            << endl;
    }
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
  sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize[2] = 1 * 1 * m_gpu_gridsize[2];
  gridsize[1] = 1;
  gridsize[0] = 1;
  blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize[1] = 1;
  blocksize[0] = 1;

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
        /*
DPCT1049:257: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ie_m_ppadding_jb_kb_true_ct0 =
                &u(1, ie - (2 * m_ppadding - 1), jb, kb, true);
            auto u_ib_m_ppadding_jb_kb_true_ct1 =
                &u(1, ib + m_ppadding, jb, kb, true);
            auto dev_SideEdge_Send_g_idx_up_ct2 = &dev_SideEdge_Send[g][idx_up];
            auto dev_SideEdge_Send_g_idx_down_ct3 =
                &dev_SideEdge_Send[g][idx_down];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[0];
            auto m_neighbor_ct9 = m_neighbor[1];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                BufferToHaloKernelY_dev_rev(
                                    u_ie_m_ppadding_jb_kb_true_ct0,
                                    u_ib_m_ppadding_jb_kb_true_ct1,
                                    dev_SideEdge_Send_g_idx_up_ct2,
                                    dev_SideEdge_Send_g_idx_down_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      else
        /*
DPCT1049:258: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ie_m_ppadding_jb_kb_true_ct0 =
                &u(1, ie - (2 * m_ppadding - 1), jb, kb, true);
            auto u_ib_m_ppadding_jb_kb_true_ct1 =
                &u(1, ib + m_ppadding, jb, kb, true);
            auto dev_SideEdge_Send_g_idx_up_ct2 = &dev_SideEdge_Send[g][idx_up];
            auto dev_SideEdge_Send_g_idx_down_ct3 =
                &dev_SideEdge_Send[g][idx_down];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[0];
            auto m_neighbor_ct9 = m_neighbor[1];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                BufferToHaloKernelY_dev(
                                    u_ie_m_ppadding_jb_kb_true_ct0,
                                    u_ib_m_ppadding_jb_kb_true_ct1,
                                    dev_SideEdge_Send_g_idx_up_ct2,
                                    dev_SideEdge_Send_g_idx_down_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      /*
DPCT1010:259: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
      CheckCudaCall(0, "BufferToHaloKernel<<<,>>>(...)", __FILE__, __LINE__);
    }
}
//-----------------------------------------------------------------------

void EW::pack_HaloArrayCU_Y( Sarray& u, int g , int st)
{
  REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
	    << " nc = " << u.m_nc );
  int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
  MPI_Status status;
  sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize[2] = 1 * 1 * m_gpu_gridsize[2];
  gridsize[1] = 1;
  gridsize[0] = 1;
  blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize[1] = 1;
  blocksize[0] = 1;

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
        /*
DPCT1049:260: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ib_jb_m_ppadding_kb_true_ct0 =
                &u(1, ib, jb + m_ppadding, kb, true);
            auto u_ib_je_m_ppadding_kb_true_ct1 =
                &u(1, ib, je - (2 * m_ppadding - 1), kb, true);
            auto dev_SideEdge_Send_g_idx_left_ct2 =
                &dev_SideEdge_Send[g][idx_left];
            auto dev_SideEdge_Send_g_idx_right_ct3 =
                &dev_SideEdge_Send[g][idx_right];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[2];
            auto m_neighbor_ct9 = m_neighbor[3];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                BufferToHaloKernelX_dev_rev(
                                    u_ib_jb_m_ppadding_kb_true_ct0,
                                    u_ib_je_m_ppadding_kb_true_ct1,
                                    dev_SideEdge_Send_g_idx_left_ct2,
                                    dev_SideEdge_Send_g_idx_right_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      else
        /*
DPCT1049:261: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ib_jb_m_ppadding_kb_true_ct0 =
                &u(1, ib, jb + m_ppadding, kb, true);
            auto u_ib_je_m_ppadding_kb_true_ct1 =
                &u(1, ib, je - (2 * m_ppadding - 1), kb, true);
            auto dev_SideEdge_Send_g_idx_left_ct2 =
                &dev_SideEdge_Send[g][idx_left];
            auto dev_SideEdge_Send_g_idx_right_ct3 =
                &dev_SideEdge_Send[g][idx_right];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[2];
            auto m_neighbor_ct9 = m_neighbor[3];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                BufferToHaloKernelX_dev(
                                    u_ib_jb_m_ppadding_kb_true_ct0,
                                    u_ib_je_m_ppadding_kb_true_ct1,
                                    dev_SideEdge_Send_g_idx_left_ct2,
                                    dev_SideEdge_Send_g_idx_right_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      /*
DPCT1010:262: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
      CheckCudaCall(0, "BufferToHaloKernel<<<,>>>(...)", __FILE__, __LINE__);
    }
}
//-----------------------------------------------------------------------

void EW::unpack_HaloArrayCU_X( Sarray& u, int g , int st)
{
  REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
	    << " nc = " << u.m_nc );
  int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
  MPI_Status status;
  sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize[2] = 1 * 1 * m_gpu_gridsize[2];
  gridsize[1] = 1;
  gridsize[0] = 1;
  blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize[1] = 1;
  blocksize[0] = 1;

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
        /*
DPCT1049:263: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ie_m_ppadding_jb_kb_true_ct0 =
                &u(1, ie - (m_ppadding - 1), jb, kb, true);
            auto u_ib_jb_kb_true_ct1 = &u(1, ib, jb, kb, true);
            auto dev_SideEdge_Recv_g_idx_up_ct2 = &dev_SideEdge_Recv[g][idx_up];
            auto dev_SideEdge_Recv_g_idx_down_ct3 =
                &dev_SideEdge_Recv[g][idx_down];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[0];
            auto m_neighbor_ct9 = m_neighbor[1];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                HaloToBufferKernelY_dev_rev(
                                    u_ie_m_ppadding_jb_kb_true_ct0,
                                    u_ib_jb_kb_true_ct1,
                                    dev_SideEdge_Recv_g_idx_up_ct2,
                                    dev_SideEdge_Recv_g_idx_down_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      else
        /*
DPCT1049:264: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ie_m_ppadding_jb_kb_true_ct0 =
                &u(1, ie - (m_ppadding - 1), jb, kb, true);
            auto u_ib_jb_kb_true_ct1 = &u(1, ib, jb, kb, true);
            auto dev_SideEdge_Recv_g_idx_up_ct2 = &dev_SideEdge_Recv[g][idx_up];
            auto dev_SideEdge_Recv_g_idx_down_ct3 =
                &dev_SideEdge_Recv[g][idx_down];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[0];
            auto m_neighbor_ct9 = m_neighbor[1];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                HaloToBufferKernelY_dev(
                                    u_ie_m_ppadding_jb_kb_true_ct0,
                                    u_ib_jb_kb_true_ct1,
                                    dev_SideEdge_Recv_g_idx_up_ct2,
                                    dev_SideEdge_Recv_g_idx_down_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      /*
DPCT1010:265: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
      CheckCudaCall(0, "HaloToBufferKernel<<<,>>>(...)", __FILE__, __LINE__);
    }
}
//-----------------------------------------------------------------------

void EW::unpack_HaloArrayCU_Y( Sarray& u, int g , int st)
{
  REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
	    << " nc = " << u.m_nc );
  int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
  MPI_Status status;
  sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
  //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
  gridsize[2] = 1 * 1 * m_gpu_gridsize[2];
  gridsize[1] = 1;
  gridsize[0] = 1;
  blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
  blocksize[1] = 1;
  blocksize[0] = 1;

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
        /*
DPCT1049:266: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ib_jb_kb_true_ct0 = &u(1, ib, jb, kb, true);
            auto u_ib_je_m_ppadding_kb_true_ct1 =
                &u(1, ib, je - (m_ppadding - 1), kb, true);
            auto dev_SideEdge_Recv_g_idx_left_ct2 =
                &dev_SideEdge_Recv[g][idx_left];
            auto dev_SideEdge_Recv_g_idx_right_ct3 =
                &dev_SideEdge_Recv[g][idx_right];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[2];
            auto m_neighbor_ct9 = m_neighbor[3];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                HaloToBufferKernelX_dev_rev(
                                    u_ib_jb_kb_true_ct0,
                                    u_ib_je_m_ppadding_kb_true_ct1,
                                    dev_SideEdge_Recv_g_idx_left_ct2,
                                    dev_SideEdge_Recv_g_idx_right_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      else
        /*
DPCT1049:267: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
         m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
            auto u_ib_jb_kb_true_ct0 = &u(1, ib, jb, kb, true);
            auto u_ib_je_m_ppadding_kb_true_ct1 =
                &u(1, ib, je - (m_ppadding - 1), kb, true);
            auto dev_SideEdge_Recv_g_idx_left_ct2 =
                &dev_SideEdge_Recv[g][idx_left];
            auto dev_SideEdge_Recv_g_idx_right_ct3 =
                &dev_SideEdge_Recv[g][idx_right];
            auto m_ppadding_ct7 = m_ppadding;
            auto m_neighbor_ct8 = m_neighbor[2];
            auto m_neighbor_ct9 = m_neighbor[3];

            cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                             [=](sycl::nd_item<3> item_ct1) {
                                HaloToBufferKernelX_dev(
                                    u_ib_jb_kb_true_ct0,
                                    u_ib_je_m_ppadding_kb_true_ct1,
                                    dev_SideEdge_Recv_g_idx_left_ct2,
                                    dev_SideEdge_Recv_g_idx_right_ct3, ni, nj,
                                    nk, m_ppadding_ct7, m_neighbor_ct8,
                                    m_neighbor_ct9, MPI_PROC_NULL, item_ct1);
                             });
         });
      /*
DPCT1010:268: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
      CheckCudaCall(0, "HaloToBufferKernel<<<,>>>(...)", __FILE__, __LINE__);
    }
}

//-----------------------------------------------------------------------

void EW::communicate_arrayCU_X( Sarray& u, int g , int st)
{
   REQUIRE2( u.m_nc == 3 || u.m_nc == 1, "Communicate array, only implemented for one- and three-component arrays"
             << " nc = " << u.m_nc );
   int ie = u.m_ie, ib=u.m_ib, je=u.m_je, jb=u.m_jb, kb=u.m_kb;//,ke=u.m_ke;
   MPI_Status status;
   int retcode;
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[2] = 1 * 1 * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;

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

      /*
DPCT1003:269: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
      SafeCudaCall((m_cuobj->m_stream[st]->wait(), 0));

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
        cudaMemcpyAsync(&m_SideEdge_Send[g][idx_up], &dev_SideEdge_Send[g][idx_up],
                        n_m_ppadding1*sizeof(float_sw4), cudaMemcpyDeviceToHost, m_cuobj->m_stream[st]);

      if (m_neighbor[0] != MPI_PROC_NULL)
        cudaMemcpyAsync(&m_SideEdge_Send[g][idx_down], &dev_SideEdge_Send[g][idx_down],
                        n_m_ppadding1*sizeof(float_sw4), cudaMemcpyDeviceToHost, m_cuobj->m_stream[st]);
      retcode = cudaStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != cudaSuccess )
        {
          cout << "Error communicate_array cudaMemcpy returned (DeviceToHost) "
               << cudaGetErrorString(retcode) << endl;
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
        cudaMemcpyAsync(&dev_SideEdge_Recv[g][idx_up], &m_SideEdge_Recv[g][idx_up],
                        n_m_ppadding1*sizeof(float_sw4), cudaMemcpyHostToDevice, m_cuobj->m_stream[st] );

      if (m_neighbor[0] != MPI_PROC_NULL)
        cudaMemcpyAsync(&dev_SideEdge_Recv[g][idx_down], &m_SideEdge_Recv[g][idx_down],
                        n_m_ppadding1*sizeof(float_sw4), cudaMemcpyHostToDevice, m_cuobj->m_stream[st] );

      retcode = cudaStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != cudaSuccess )
        {
          cout << "Error communicate_array cudaMemcpy returned (Host2Device) "
               << cudaGetErrorString(retcode) << endl;
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
   int retcode;
   sycl::range<3> gridsize(1, 1, 1), blocksize(1, 1, 1);
   //gridsize.x  = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[2] = 1 * 1 * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;

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

      /*
DPCT1003:270: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
      SafeCudaCall((m_cuobj->m_stream[st]->wait(), 0));

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
        cudaMemcpyAsync(&m_SideEdge_Send[g][idx_left], &dev_SideEdge_Send[g][idx_left],
                        n_m_ppadding2*sizeof(float_sw4), cudaMemcpyDeviceToHost, m_cuobj->m_stream[st]);

      if (m_neighbor[3] != MPI_PROC_NULL)
        cudaMemcpyAsync(&m_SideEdge_Send[g][idx_right], &dev_SideEdge_Send[g][idx_right],
                        n_m_ppadding2*sizeof(float_sw4), cudaMemcpyDeviceToHost, m_cuobj->m_stream[st]);
      
      retcode = cudaStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != cudaSuccess )
        {
          cout << "Error communicate_array cudaMemcpy returned (DeviceToHost) "
               << cudaGetErrorString(retcode) << endl;
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
        cudaMemcpyAsync(&dev_SideEdge_Recv[g][idx_left], &m_SideEdge_Recv[g][idx_left],
                        n_m_ppadding2*sizeof(float_sw4), cudaMemcpyHostToDevice, m_cuobj->m_stream[st] );

      if (m_neighbor[3] != MPI_PROC_NULL)
        cudaMemcpyAsync(&dev_SideEdge_Recv[g][idx_right], &m_SideEdge_Recv[g][idx_right],
                        n_m_ppadding2*sizeof(float_sw4), cudaMemcpyHostToDevice, m_cuobj->m_stream[st] );
      
      retcode = cudaStreamSynchronize(m_cuobj->m_stream[st]);
      if( retcode != cudaSuccess )
        {
          cout << "Error communicate_array cudaMemcpy returned (Host2Device) "
               << cudaGetErrorString(retcode) << endl;
          exit(1);
        }

#endif
   }
}

//-----------------------------------------------------------------------

void EW::setup_device_communication_array() try {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
  dev_SideEdge_Send.resize(mNumberOfGrids);
  dev_SideEdge_Recv.resize(mNumberOfGrids);
#ifndef SW4_CUDA_AWARE_MPI
  m_SideEdge_Send.resize(mNumberOfGrids);
  m_SideEdge_Recv.resize(mNumberOfGrids);
#endif

  if( m_ndevice > 0 )
  {
     int retcode;

     for( int g=0 ; g<mNumberOfGrids; g++)
     {

        int ni = m_iEnd[g] - m_iStart[g] + 1;
        int nj = m_jEnd[g] - m_jStart[g] + 1;
        int nk = m_kEnd[g] - m_kStart[g] + 1;
        int n_m_ppadding1 = 3*nj*nk*m_ppadding;
        int n_m_ppadding2 = 3*ni*nk*m_ppadding;

        /*
DPCT1003:275: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode = (dev_SideEdge_Send[g] = (float *)sycl::malloc_device(
                       sizeof(float_sw4) * 2 * (n_m_ppadding1 + n_m_ppadding2),
                       dpct::get_default_queue()),
                   0);
        /*
DPCT1000:272: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:271: The statement could not be removed.
*/
           cout
               << "Error, EW::setup_device_communication_arra cudaMalloc "
                  "returned "
               /*
               DPCT1009:276: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;

        /*
DPCT1003:277: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
        retcode = (dev_SideEdge_Recv[g] = (float *)sycl::malloc_device(
                       sizeof(float_sw4) * 2 * (n_m_ppadding1 + n_m_ppadding2),
                       dpct::get_default_queue()),
                   0);
        /*
DPCT1000:274: Error handling if-stmt was detected but could not be rewritten.
*/
        if (retcode != 0)
           /*
DPCT1001:273: The statement could not be removed.
*/
           cout
               << "Error, EW::setup_device_communication_arra cudaMalloc "
                  "returned "
               /*
               DPCT1009:278: SYCL uses exceptions to report errors and does not
               use the error codes. The original code was commented out and a
               warning string was inserted. You need to rewrite this code.
               */
               << "cudaGetErrorString not supported" /*cudaGetErrorString(retcode)*/
               << endl;

#ifndef SW4_CUDA_AWARE_MPI

        retcode = cudaMallocHost( (void**)&m_SideEdge_Send[g], sizeof(float_sw4)*2*(n_m_ppadding1+n_m_ppadding2) );
        if( retcode != cudaSuccess )
           cout << "Error, EW::setup_device_communication_arra cudaMallocHost returned "
                << cudaGetErrorString(retcode) << endl;

        retcode = cudaMallocHost( (void**)&m_SideEdge_Recv[g], sizeof(float_sw4)*2*(n_m_ppadding1+n_m_ppadding2) );
        if( retcode != cudaSuccess )
           cout << "Error, EW::setup_device_communication_arra cudaMallocHost returned "
                << cudaGetErrorString(retcode) << endl;

#endif

     }
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
void EW::allocateTimeSeriesOnDeviceCU(int &nvals, int &ntloc, int *&i0dev,
                                      int *&j0dev, int *&k0dev, int *&g0dev,
                                      int *&modedev, float_sw4 **&urec_dev,
                                      float_sw4 **&urec_host,
                                      float_sw4 **&urec_hdev) try {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  int retval;
  if( ntloc > 0 )
  {
  // Allocate memory on device, and copy the location to vectors on device
     /*
DPCT1003:303: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval =
         (i0dev = sycl::malloc_device<int>(ntloc, dpct::get_default_queue()),
          0);
     /*
DPCT1000:280: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:279: The statement could not be removed.
*/
        /*
DPCT1009:304: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMalloc of i0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:305: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval =
         (j0dev = sycl::malloc_device<int>(ntloc, dpct::get_default_queue()),
          0);
     /*
DPCT1000:282: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:281: The statement could not be removed.
*/
        /*
DPCT1009:306: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMalloc of j0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:307: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval =
         (k0dev = sycl::malloc_device<int>(ntloc, dpct::get_default_queue()),
          0);
     /*
DPCT1000:284: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:283: The statement could not be removed.
*/
        /*
DPCT1009:308: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMalloc of k0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:309: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval =
         (g0dev = sycl::malloc_device<int>(ntloc, dpct::get_default_queue()),
          0);
     /*
DPCT1000:286: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:285: The statement could not be removed.
*/
        /*
DPCT1009:310: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMalloc of g0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:311: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval =
         (modedev = sycl::malloc_device<int>(ntloc, dpct::get_default_queue()),
          0);
     /*
DPCT1000:288: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:287: The statement could not be removed.
*/
        /*
DPCT1009:312: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMalloc of modedev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:313: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval = (dpct::get_default_queue()
                   .memcpy(i0dev, &i0vect[0], sizeof(int) * ntloc)
                   .wait(),
               0);
     /*
DPCT1000:290: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:289: The statement could not be removed.
*/
        /*
DPCT1009:314: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMmemcpy of i0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:315: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval = (dpct::get_default_queue()
                   .memcpy(j0dev, &j0vect[0], sizeof(int) * ntloc)
                   .wait(),
               0);
     /*
DPCT1000:292: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:291: The statement could not be removed.
*/
        /*
DPCT1009:316: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMmemcpy of j0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:317: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval = (dpct::get_default_queue()
                   .memcpy(k0dev, &k0vect[0], sizeof(int) * ntloc)
                   .wait(),
               0);
     /*
DPCT1000:294: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:293: The statement could not be removed.
*/
        /*
DPCT1009:318: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMmemcpy of k0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:319: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval = (dpct::get_default_queue()
                   .memcpy(g0dev, &g0vect[0], sizeof(int) * ntloc)
                   .wait(),
               0);
     /*
DPCT1000:296: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:295: The statement could not be removed.
*/
        /*
DPCT1009:320: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMmemcpy of g0dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:321: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval = (dpct::get_default_queue()
                   .memcpy(modedev, &modevect[0], sizeof(int) * ntloc)
                   .wait(),
               0);
     /*
DPCT1000:298: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:297: The statement could not be removed.
*/
        /*
DPCT1009:322: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMmemcpy of modedev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;

    // Allocate memory on host and and device to hold the data 
     float_sw4* devmem;
     /*
DPCT1003:323: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval =
         (devmem = sycl::malloc_device<float>(nvals, dpct::get_default_queue()),
          0);
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
     /*
DPCT1003:324: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval = (urec_dev = sycl::malloc_device<float *>(
                   ntloc, dpct::get_default_queue()),
               0);
     /*
DPCT1000:300: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:299: The statement could not be removed.
*/
        /*
DPCT1009:325: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMalloc of urec_dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
     /*
DPCT1003:326: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
     retval = (dpct::get_default_queue()
                   .memcpy(urec_dev, urec_hdev, sizeof(float_sw4 *) * ntloc)
                   .wait(),
               0);
     /*
DPCT1000:302: Error handling if-stmt was detected but could not be rewritten.
*/
     if (retval != 0)
        /*
DPCT1001:301: The statement could not be removed.
*/
        /*
DPCT1009:327: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
        cout
            << "Error in cudaMmemcpy of urec_dev retval = "
            << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
            << endl;
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//-----------------------------------------------------------------------
void EW::extractRecordDataCU(int nt, int *mode, int *i0v, int *j0v, int *k0v,
                             int *g0v, float_sw4 **urec_dev, Sarray *dev_Um,
                             Sarray *dev_U, float_sw4 dt, float_sw4 *h_dev,
                             Sarray *dev_metric, Sarray *dev_j, int st,
                             int nvals, float_sw4 *urec_hostmem,
                             float_sw4 *urec_devmem) try {
   sycl::range<3> blocksize(1, 1, 1), gridsize(1, 1, 1);
   gridsize[2] = m_gpu_gridsize[0] * m_gpu_gridsize[1] * m_gpu_gridsize[2];
   gridsize[1] = 1;
   gridsize[0] = 1;
   blocksize[2] = m_gpu_blocksize[0] * m_gpu_blocksize[1] * m_gpu_blocksize[2];
   blocksize[1] = 1;
   blocksize[0] = 1;
   /*
DPCT1049:328: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
   m_cuobj->m_stream[st]->submit([&](sycl::handler &cgh) {
      auto mNumberOfCartesianGrids_ct11 = mNumberOfCartesianGrids;

      cgh.parallel_for(sycl::nd_range<3>(gridsize * blocksize, blocksize),
                       [=](sycl::nd_item<3> item_ct1) {
                          extractRecordData_dev(
                              nt, mode, i0v, j0v, k0v, g0v, urec_dev, dev_Um,
                              dev_U, dt, h_dev, mNumberOfCartesianGrids_ct11,
                              dev_metric, dev_j, item_ct1);
                       });
   });
   /*
DPCT1003:331: Migrated API does not return error code. (*, 0) is inserted. You
may need to rewrite this code.
*/
   int retval =
       (dpct::get_default_queue()
            .memcpy(urec_hostmem, urec_devmem, sizeof(float_sw4) * nvals)
            .wait(),
        0);
   /*
DPCT1000:330: Error handling if-stmt was detected but could not be rewritten.
*/
   if (retval != 0)
      /*
DPCT1001:329: The statement could not be removed.
*/
      /*
DPCT1009:332: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
      cout << "Error in cudaMemcpy in EW::extractRecordDataCU retval = "
           << "cudaGetErrorString not supported" /*cudaGetErrorString(retval)*/
           << endl;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
