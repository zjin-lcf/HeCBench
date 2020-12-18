#ifndef _Vector_functions_hpp_
#define _Vector_functions_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// ************************************************************************
//@HEADER

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <sstream>
#include <fstream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef MINIFE_HAVE_TBB
#include <LockingVector.hpp>
#endif

#include <TypeTraits.hpp>
#include <Vector.hpp>

#define MINIFE_MIN(X, Y)  ((X) < (Y) ? (X) : (Y))

namespace miniFE {


  template<typename VectorType>
    void write_vector(const std::string& filename,
        const VectorType& vec)
    {
      int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

      std::ostringstream osstr;
      osstr << filename << "." << numprocs << "." << myproc;
      std::string full_name = osstr.str();
      std::ofstream ofs(full_name.c_str());

      typedef typename VectorType::ScalarType ScalarType;

      const std::vector<ScalarType>& coefs = vec.coefs;
      for(int p=0; p<numprocs; ++p) {
        if (p == myproc) {
          if (p == 0) {
            ofs << vec.local_size << std::endl;
          }

          typename VectorType::GlobalOrdinalType first = vec.startIndex;
          for(size_t i=0; i<vec.local_size; ++i) {
            ofs << first+i << " " << coefs[i] << std::endl;
          }
        }
#ifdef HAVE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
      }
    }

  template<typename VectorType>
    void sum_into_vector(size_t num_indices,
        const typename VectorType::GlobalOrdinalType* indices,
        const typename VectorType::ScalarType* coefs,
        VectorType& vec)
    {
      typedef typename VectorType::GlobalOrdinalType GlobalOrdinal;
      typedef typename VectorType::ScalarType Scalar;

      GlobalOrdinal first = vec.startIndex;
      GlobalOrdinal last = first + vec.local_size - 1;

      std::vector<Scalar>& vec_coefs = vec.coefs;

      for(size_t i=0; i<num_indices; ++i) {
        if (indices[i] < first || indices[i] > last) continue;
        size_t idx = indices[i] - first;

#pragma omp atomic
        vec_coefs[idx] += coefs[i];
      }
    }

#ifdef MINIFE_HAVE_TBB
  template<typename VectorType>
    void sum_into_vector(size_t num_indices,
        const typename VectorType::GlobalOrdinalType* indices,
        const typename VectorType::ScalarType* coefs,
        LockingVector<VectorType>& vec)
    {
      vec.sum_in(num_indices, indices, coefs);
    }
#endif

  //------------------------------------------------------------
  //Compute the update of a vector with the sum of two scaled vectors where:
  //
  // w = alpha*x + beta*y
  //
  // x,y - input vectors
  //
  // alpha,beta - scalars applied to x and y respectively
  //
  // w - output vector
  //
  template <typename VectorType> 
    void waxby_kernel(
        const int n,
        const typename VectorType::ScalarType alpha, 
        const typename VectorType::ScalarType *xcoefs, 
        const typename VectorType::ScalarType beta,
        const typename VectorType::ScalarType *ycoefs, 
        typename VectorType::ScalarType *wcoefs,
        sycl::nd_item<3> item_ct1) 
    {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) wcoefs[idx] = alpha * xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType> 
    void yaxby_kernel(
        const int n,
        const typename VectorType::ScalarType alpha, 
        const typename VectorType::ScalarType *xcoefs, 
        const typename VectorType::ScalarType beta,
        typename VectorType::ScalarType *ycoefs ,
        sycl::nd_item<3> item_ct1)
    {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) ycoefs[idx] = alpha * xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType> 
    void wxby_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs, 
        const typename VectorType::ScalarType beta,
        const typename VectorType::ScalarType *ycoefs, 
        typename VectorType::ScalarType *wcoefs,
        sycl::nd_item<3> item_ct1) 
    {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) wcoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType> 
    void yxby_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs, 
        const typename VectorType::ScalarType beta,
        typename VectorType::ScalarType *ycoefs,
        sycl::nd_item<3> item_ct1)
    {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) ycoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType> 
    void wx_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs, 
        typename VectorType::ScalarType *wcoefs,
        sycl::nd_item<3> item_ct1) 
    {

  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) wcoefs[idx] = xcoefs[idx];
    }

  template <typename VectorType> 
    void dyx_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs, 
        typename VectorType::ScalarType *ycoefs,
        sycl::nd_item<3> item_ct1) 
    {

  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) ycoefs[idx] += xcoefs[idx];
    }

  template <typename VectorType> 
    void wax_kernel(
        const int n,
        const typename VectorType::ScalarType alpha, 
        const typename VectorType::ScalarType *xcoefs, 
        typename VectorType::ScalarType *wcoefs,
        sycl::nd_item<3> item_ct1) 
    {

  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) wcoefs[idx] = alpha * xcoefs[idx];
    }

  template <typename VectorType> 
    void dyax_kernel(
        const int n,
        const typename VectorType::ScalarType alpha, 
        const typename VectorType::ScalarType *xcoefs, 
        typename VectorType::ScalarType *ycoefs,
        sycl::nd_item<3> item_ct1) 
    {

  int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
      if (idx<n) ycoefs[idx] += alpha * xcoefs[idx];
    }
  //------------------------------------------------------------
  //Compute the update of a vector with the sum of two scaled vectors where:
  //
  // w = alpha*x + beta*y
  //
  // x,y - input vectors
  //
  // alpha,beta - scalars applied to x and y respectively
  //
  // w - output vector
  //
  template<typename VectorType>
    void
    waxpby(typename VectorType::ScalarType alpha, const VectorType& x,
        typename VectorType::ScalarType beta, const VectorType& y,
        VectorType& w,
        typename VectorType::ScalarType *d_xcoefs,
        typename VectorType::ScalarType *d_ycoefs,
        typename VectorType::ScalarType *d_wcoefs)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
#ifdef MINIFE_DEBUG
      std::cout << "Starting WAXPBY..." << std::endl;
#endif

#ifdef MINIFE_DEBUG
      if (y.local_size < x.local_size || w.local_size < x.local_size) {
        std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x." << std::endl;
        return;
      }
#endif


      const int n = x.coefs.size();
  sycl::range<3> grids((n + 255) / 256, 1, 1);
  sycl::range<3> threads(256, 1, 1);

      if(beta == 0.0) {
        if(alpha == 1.0) {
      std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct1 =
          dpct::get_buffer_and_offset(d_xcoefs);
      size_t d_xcoefs_offset_ct1 = d_xcoefs_buf_ct1.second;
      std::pair<dpct::buffer_t, size_t> d_wcoefs_buf_ct2 =
          dpct::get_buffer_and_offset(d_wcoefs);
      size_t d_wcoefs_offset_ct2 = d_wcoefs_buf_ct2.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_xcoefs_acc_ct1 =
            d_xcoefs_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_wcoefs_acc_ct2 =
            d_wcoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = grids * threads;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              typename VectorType::ScalarType *d_xcoefs_ct1 =
                  (typename VectorType::ScalarType *)(&d_xcoefs_acc_ct1[0] +
                                                      d_xcoefs_offset_ct1);
              typename VectorType::ScalarType *d_wcoefs_ct2 =
                  (typename VectorType::ScalarType *)(&d_wcoefs_acc_ct2[0] +
                                                      d_wcoefs_offset_ct2);
              wx_kernel<VectorType>(n, d_xcoefs_ct1, d_wcoefs_ct2, item_ct1);
            });
      });
        } else {
      std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct2 =
          dpct::get_buffer_and_offset(d_xcoefs);
      size_t d_xcoefs_offset_ct2 = d_xcoefs_buf_ct2.second;
      std::pair<dpct::buffer_t, size_t> d_wcoefs_buf_ct3 =
          dpct::get_buffer_and_offset(d_wcoefs);
      size_t d_wcoefs_offset_ct3 = d_wcoefs_buf_ct3.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_xcoefs_acc_ct2 =
            d_xcoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_wcoefs_acc_ct3 =
            d_wcoefs_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = grids * threads;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              typename VectorType::ScalarType *d_xcoefs_ct2 =
                  (typename VectorType::ScalarType *)(&d_xcoefs_acc_ct2[0] +
                                                      d_xcoefs_offset_ct2);
              typename VectorType::ScalarType *d_wcoefs_ct3 =
                  (typename VectorType::ScalarType *)(&d_wcoefs_acc_ct3[0] +
                                                      d_wcoefs_offset_ct3);
              wax_kernel<VectorType>(n, alpha, d_xcoefs_ct2, d_wcoefs_ct3,
                                     item_ct1);
            });
      });
        }
      } else {
        if(alpha == 1.0) {
      std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct1 =
          dpct::get_buffer_and_offset(d_xcoefs);
      size_t d_xcoefs_offset_ct1 = d_xcoefs_buf_ct1.second;
      std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct3 =
          dpct::get_buffer_and_offset(d_ycoefs);
      size_t d_ycoefs_offset_ct3 = d_ycoefs_buf_ct3.second;
      std::pair<dpct::buffer_t, size_t> d_wcoefs_buf_ct4 =
          dpct::get_buffer_and_offset(d_wcoefs);
      size_t d_wcoefs_offset_ct4 = d_wcoefs_buf_ct4.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_xcoefs_acc_ct1 =
            d_xcoefs_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_ycoefs_acc_ct3 =
            d_ycoefs_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_wcoefs_acc_ct4 =
            d_wcoefs_buf_ct4.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = grids * threads;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              typename VectorType::ScalarType *d_xcoefs_ct1 =
                  (typename VectorType::ScalarType *)(&d_xcoefs_acc_ct1[0] +
                                                      d_xcoefs_offset_ct1);
              typename VectorType::ScalarType *d_ycoefs_ct3 =
                  (typename VectorType::ScalarType *)(&d_ycoefs_acc_ct3[0] +
                                                      d_ycoefs_offset_ct3);
              typename VectorType::ScalarType *d_wcoefs_ct4 =
                  (typename VectorType::ScalarType *)(&d_wcoefs_acc_ct4[0] +
                                                      d_wcoefs_offset_ct4);
              wxby_kernel<VectorType>(n, d_xcoefs_ct1, beta, d_ycoefs_ct3,
                                      d_wcoefs_ct4, item_ct1);
            });
      });
        } else {
      std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct2 =
          dpct::get_buffer_and_offset(d_xcoefs);
      size_t d_xcoefs_offset_ct2 = d_xcoefs_buf_ct2.second;
      std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct4 =
          dpct::get_buffer_and_offset(d_ycoefs);
      size_t d_ycoefs_offset_ct4 = d_ycoefs_buf_ct4.second;
      std::pair<dpct::buffer_t, size_t> d_wcoefs_buf_ct5 =
          dpct::get_buffer_and_offset(d_wcoefs);
      size_t d_wcoefs_offset_ct5 = d_wcoefs_buf_ct5.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_xcoefs_acc_ct2 =
            d_xcoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_ycoefs_acc_ct4 =
            d_ycoefs_buf_ct4.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_wcoefs_acc_ct5 =
            d_wcoefs_buf_ct5.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = grids * threads;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              typename VectorType::ScalarType *d_xcoefs_ct2 =
                  (typename VectorType::ScalarType *)(&d_xcoefs_acc_ct2[0] +
                                                      d_xcoefs_offset_ct2);
              typename VectorType::ScalarType *d_ycoefs_ct4 =
                  (typename VectorType::ScalarType *)(&d_ycoefs_acc_ct4[0] +
                                                      d_ycoefs_offset_ct4);
              typename VectorType::ScalarType *d_wcoefs_ct5 =
                  (typename VectorType::ScalarType *)(&d_wcoefs_acc_ct5[0] +
                                                      d_wcoefs_offset_ct5);
              waxby_kernel<VectorType>(n, alpha, d_xcoefs_ct2, beta,
                                       d_ycoefs_ct4, d_wcoefs_ct5, item_ct1);
            });
      });
        }
      }

#ifdef MINIFE_DEBUG
      std::cout << "Finished WAXPBY." << std::endl;
#endif
    }

  template<typename VectorType>
    void
    daxpby(const MINIFE_SCALAR alpha, 
        const VectorType& x,
        const MINIFE_SCALAR beta, 
        VectorType& y,
        MINIFE_SCALAR *d_xcoefs,
        MINIFE_SCALAR *d_ycoefs)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

      const MINIFE_LOCAL_ORDINAL n = MINIFE_MIN(x.coefs.size(), y.coefs.size());

  sycl::range<3> grids((n + 255) / 256, 1, 1);
  sycl::range<3> threads(256, 1, 1);

      if(alpha == 1.0 && beta == 1.0) {
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct1 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct1 = d_xcoefs_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct2 =
        dpct::get_buffer_and_offset(d_ycoefs);
    size_t d_ycoefs_offset_ct2 = d_ycoefs_buf_ct2.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_xcoefs_acc_ct1 =
          d_xcoefs_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_ycoefs_acc_ct2 =
          d_ycoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            double *d_xcoefs_ct1 =
                (double *)(&d_xcoefs_acc_ct1[0] + d_xcoefs_offset_ct1);
            double *d_ycoefs_ct2 =
                (double *)(&d_ycoefs_acc_ct2[0] + d_ycoefs_offset_ct2);
            dyx_kernel<VectorType>(n, d_xcoefs_ct1, d_ycoefs_ct2, item_ct1);
          });
    });
      } else if (beta == 1.0) {
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct2 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct2 = d_xcoefs_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct3 =
        dpct::get_buffer_and_offset(d_ycoefs);
    size_t d_ycoefs_offset_ct3 = d_ycoefs_buf_ct3.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_xcoefs_acc_ct2 =
          d_xcoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_ycoefs_acc_ct3 =
          d_ycoefs_buf_ct3.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            double *d_xcoefs_ct2 =
                (double *)(&d_xcoefs_acc_ct2[0] + d_xcoefs_offset_ct2);
            double *d_ycoefs_ct3 =
                (double *)(&d_ycoefs_acc_ct3[0] + d_ycoefs_offset_ct3);
            dyax_kernel<VectorType>(n, alpha, d_xcoefs_ct2, d_ycoefs_ct3,
                                    item_ct1);
          });
    });
      } else if (alpha == 1.0) {
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct1 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct1 = d_xcoefs_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct3 =
        dpct::get_buffer_and_offset(d_ycoefs);
    size_t d_ycoefs_offset_ct3 = d_ycoefs_buf_ct3.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_xcoefs_acc_ct1 =
          d_xcoefs_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_ycoefs_acc_ct3 =
          d_ycoefs_buf_ct3.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            double *d_xcoefs_ct1 =
                (double *)(&d_xcoefs_acc_ct1[0] + d_xcoefs_offset_ct1);
            double *d_ycoefs_ct3 =
                (double *)(&d_ycoefs_acc_ct3[0] + d_ycoefs_offset_ct3);
            yxby_kernel<VectorType>(n, d_xcoefs_ct1, beta, d_ycoefs_ct3,
                                    item_ct1);
          });
    });
      } else if (beta == 0.0) {
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct2 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct2 = d_xcoefs_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct3 =
        dpct::get_buffer_and_offset(d_ycoefs);
    size_t d_ycoefs_offset_ct3 = d_ycoefs_buf_ct3.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_xcoefs_acc_ct2 =
          d_xcoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_ycoefs_acc_ct3 =
          d_ycoefs_buf_ct3.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            double *d_xcoefs_ct2 =
                (double *)(&d_xcoefs_acc_ct2[0] + d_xcoefs_offset_ct2);
            double *d_ycoefs_ct3 =
                (double *)(&d_ycoefs_acc_ct3[0] + d_ycoefs_offset_ct3);
            wax_kernel<VectorType>(n, alpha, d_xcoefs_ct2, d_ycoefs_ct3,
                                   item_ct1);
          });
    }); // reuse the "wax" kernel
      } else {
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct2 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct2 = d_xcoefs_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct4 =
        dpct::get_buffer_and_offset(d_ycoefs);
    size_t d_ycoefs_offset_ct4 = d_ycoefs_buf_ct4.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_xcoefs_acc_ct2 =
          d_xcoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_ycoefs_acc_ct4 =
          d_ycoefs_buf_ct4.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            double *d_xcoefs_ct2 =
                (double *)(&d_xcoefs_acc_ct2[0] + d_xcoefs_offset_ct2);
            double *d_ycoefs_ct4 =
                (double *)(&d_ycoefs_acc_ct4[0] + d_ycoefs_offset_ct4);
            yaxby_kernel<VectorType>(n, alpha, d_xcoefs_ct2, beta, d_ycoefs_ct4,
                                     item_ct1);
          });
    });
      }

    }

  template<typename Scalar>
    void dot_kernel(const MINIFE_LOCAL_ORDINAL n, 
        const Scalar* x, 
        const Scalar* y, 
              Scalar* d,
              sycl::nd_item<3> item_ct1,
              Scalar *red) 
    {
      Scalar sum=0;
  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
       idx < n;
       idx += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
        sum+=x[idx]*y[idx];
      }

      //Do a shared memory reduction on the dot product

  red[item_ct1.get_local_id(2)] = sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < n) {
      sum += red[item_ct1.get_local_id(2) + n];
      red[item_ct1.get_local_id(2)] = sum;
    }
      }

      //save partial dot products
    if (item_ct1.get_local_id(2) == 0) d[item_ct1.get_group(2)] = sum;
    }

  template<typename Scalar>
    void final_reduce(Scalar *d, sycl::nd_item<3> item_ct1, Scalar *red) {
  Scalar sum = d[item_ct1.get_local_id(2)];

  red[item_ct1.get_local_id(2)] = sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < n) {
      sum += red[item_ct1.get_local_id(2) + n];
      red[item_ct1.get_local_id(2)] = sum;
    }
      }
      //save final dot product at the front
    if (item_ct1.get_local_id(2) == 0) d[0] = sum;
    }


  //-----------------------------------------------------------
  //Compute the dot product of two vectors where:
  //
  // x,y - input vectors
  //
  // result - return-value
  //
  template<typename Vector>
    typename TypeTraits<typename Vector::ScalarType>::magnitude_type
    dot(const Vector& x,
        const Vector& y,
        typename Vector::ScalarType *d_xcoefs,
        typename Vector::ScalarType *d_ycoefs)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
      const MINIFE_LOCAL_ORDINAL n = x.coefs.size();
      typedef typename Vector::ScalarType Scalar;
      Scalar result = 0;
      int BLOCK_SIZE = 256;
  int NUM_BLOCKS = std::min(1024, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sycl::range<3> grids(NUM_BLOCKS, 1, 1);
  sycl::range<3> threads(BLOCK_SIZE, 1, 1);
      Scalar* d;
  dpct::dpct_malloc((void **)&d, sizeof(Scalar) * 1024);
  dpct::dpct_memset(d, 0, sizeof(Scalar) * 1024);
  {
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct1 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct1 = d_xcoefs_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_ycoefs_buf_ct2 =
        dpct::get_buffer_and_offset(d_ycoefs);
    size_t d_ycoefs_offset_ct2 = d_ycoefs_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_buf_ct3 =
        dpct::get_buffer_and_offset(d);
    size_t d_offset_ct3 = d_buf_ct3.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<Scalar, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          red_acc_ct1(sycl::range<1>(256), cgh);
      auto d_xcoefs_acc_ct1 =
          d_xcoefs_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_ycoefs_acc_ct2 =
          d_ycoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_acc_ct3 =
          d_buf_ct3.first.get_access<sycl::access::mode::read_write>(cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            typename Vector::ScalarType *d_xcoefs_ct1 =
                (typename Vector::ScalarType *)(&d_xcoefs_acc_ct1[0] +
                                                d_xcoefs_offset_ct1);
            typename Vector::ScalarType *d_ycoefs_ct2 =
                (typename Vector::ScalarType *)(&d_ycoefs_acc_ct2[0] +
                                                d_ycoefs_offset_ct2);
            Scalar *d_ct3 = (Scalar *)(&d_acc_ct3[0] + d_offset_ct3);
            dot_kernel<Scalar>(n, d_xcoefs_ct1, d_ycoefs_ct2, d_ct3, item_ct1,
                               red_acc_ct1.get_pointer());
          });
    });
  }
  {
    std::pair<dpct::buffer_t, size_t> d_buf_ct0 =
        dpct::get_buffer_and_offset(d);
    size_t d_offset_ct0 = d_buf_ct0.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<Scalar, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          red_acc_ct1(sycl::range<1>(256), cgh);
      auto d_acc_ct0 =
          d_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
          [=](sycl::nd_item<3> item_ct1) {
            Scalar *d_ct0 = (Scalar *)(&d_acc_ct0[0] + d_offset_ct0);
            final_reduce<Scalar>(d_ct0, item_ct1, red_acc_ct1.get_pointer());
          });
    });
  }
  dpct::dpct_memcpy(&result, d, sizeof(Scalar), dpct::device_to_host);
  dpct::dpct_free(d);

#ifdef HAVE_MPI
      typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;
      magnitude local_dot = result, global_dot = 0;
      MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
      MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
      return global_dot;
#else
      return result;
#endif
    }

  template<typename Vector>
    typename TypeTraits<typename Vector::ScalarType>::magnitude_type
    dot_r2(const Vector& x, const typename Vector::ScalarType *d_xcoefs)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      std::cout << "[" << myrank << "] Starting dot..." << std::endl;
#endif
#endif

      const MINIFE_LOCAL_ORDINAL n = x.coefs.size();
      typedef typename Vector::ScalarType Scalar;
      Scalar result = 0;
      int BLOCK_SIZE = 256;
  int NUM_BLOCKS = std::min(1024, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sycl::range<3> grids(NUM_BLOCKS, 1, 1);
  sycl::range<3> threads(BLOCK_SIZE, 1, 1);
      Scalar* d;
  dpct::dpct_malloc((void **)&d, sizeof(Scalar) * 1024);
  dpct::dpct_memset(d, 0, sizeof(Scalar) * 1024);
  {
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct1 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct1 = d_xcoefs_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_xcoefs_buf_ct2 =
        dpct::get_buffer_and_offset(d_xcoefs);
    size_t d_xcoefs_offset_ct2 = d_xcoefs_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_buf_ct3 =
        dpct::get_buffer_and_offset(d);
    size_t d_offset_ct3 = d_buf_ct3.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<Scalar, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          red_acc_ct1(sycl::range<1>(256), cgh);
      auto d_xcoefs_acc_ct1 =
          d_xcoefs_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_xcoefs_acc_ct2 =
          d_xcoefs_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_acc_ct3 =
          d_buf_ct3.first.get_access<sycl::access::mode::read_write>(cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            const typename Vector::ScalarType *d_xcoefs_ct1 =
                (const typename Vector::ScalarType *)(&d_xcoefs_acc_ct1[0] +
                                                      d_xcoefs_offset_ct1);
            const typename Vector::ScalarType *d_xcoefs_ct2 =
                (const typename Vector::ScalarType *)(&d_xcoefs_acc_ct2[0] +
                                                      d_xcoefs_offset_ct2);
            Scalar *d_ct3 = (Scalar *)(&d_acc_ct3[0] + d_offset_ct3);
            dot_kernel<Scalar>(n, d_xcoefs_ct1, d_xcoefs_ct2, d_ct3, item_ct1,
                               red_acc_ct1.get_pointer());
          });
    });
  }
  {
    std::pair<dpct::buffer_t, size_t> d_buf_ct0 =
        dpct::get_buffer_and_offset(d);
    size_t d_offset_ct0 = d_buf_ct0.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<Scalar, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          red_acc_ct1(sycl::range<1>(256), cgh);
      auto d_acc_ct0 =
          d_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
          [=](sycl::nd_item<3> item_ct1) {
            Scalar *d_ct0 = (Scalar *)(&d_acc_ct0[0] + d_offset_ct0);
            final_reduce<Scalar>(d_ct0, item_ct1, red_acc_ct1.get_pointer());
          });
    });
  }
  dpct::dpct_memcpy(&result, d, sizeof(Scalar), dpct::device_to_host);
  dpct::dpct_free(d);

#ifdef HAVE_MPI
      typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;
      magnitude local_dot = result, global_dot = 0;
      MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
      MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);

#ifdef MINIFE_DEBUG
      std::cout << "[" << myrank << "] Completed dot." << std::endl;
#endif

      return global_dot;
#else
#ifdef MINIFE_DEBUG
      std::cout << "[" << 0 << "] Completed dot." << std::endl;
#endif
      return result;
#endif
    }

}//namespace miniFE

#endif

