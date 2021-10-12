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

#include <vector>
#include <sstream>
#include <fstream>
#include <CL/sycl.hpp>
#include "common.h"

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
template<typename VectorType>
void
  waxpby(typename VectorType::ScalarType alpha, const VectorType& x,
         typename VectorType::ScalarType beta, const VectorType& y,
         VectorType& w,
         queue &q,
         buffer<typename VectorType::ScalarType, 1> &d_xcoefs,
         buffer<typename VectorType::ScalarType, 1> &d_ycoefs,
         buffer<typename VectorType::ScalarType, 1> &d_wcoefs)
{
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
  range<1> gws ((n+255)/256*256);
  range<1> lws (256);

  if(beta == 0.0) {
    if(alpha == 1.0) {
      q.submit([&] (handler &h) {
        auto wcoefs = d_wcoefs.template get_access<sycl_write>(h);
        auto xcoefs = d_xcoefs.template get_access<sycl_read>(h);
        h.parallel_for<class wx_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < n) wcoefs[i] = xcoefs[i];
        });
      });
    } else {
      q.submit([&] (handler &h) {
        auto wcoefs = d_wcoefs.template get_access<sycl_write>(h);
        auto xcoefs = d_xcoefs.template get_access<sycl_read>(h);
        h.parallel_for<class wax_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < n) wcoefs[i] = alpha*xcoefs[i];
        });
      });
    }
  } else {
    if(alpha == 1.0) {
      q.submit([&] (handler &h) {
        auto wcoefs = d_wcoefs.template get_access<sycl_write>(h);
        auto xcoefs = d_xcoefs.template get_access<sycl_read>(h);
        auto ycoefs = d_ycoefs.template get_access<sycl_read>(h);
        h.parallel_for<class wxby_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < n) wcoefs[i] = xcoefs[i]+beta*ycoefs[i];
        });
      });
    } else {
      q.submit([&] (handler &h) {
        auto wcoefs = d_wcoefs.template get_access<sycl_write>(h);
        auto xcoefs = d_xcoefs.template get_access<sycl_read>(h);
        auto ycoefs = d_ycoefs.template get_access<sycl_read>(h);
        h.parallel_for<class waxby_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < n) wcoefs[i] = alpha*xcoefs[i]+beta*ycoefs[i];
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
    queue &q,
    buffer<MINIFE_SCALAR, 1> &d_xcoefs,
    buffer<MINIFE_SCALAR, 1> &d_ycoefs)
{

  const MINIFE_LOCAL_ORDINAL n = MINIFE_MIN(x.coefs.size(), y.coefs.size());

  range<1> gws ((n+255)/256*256);
  range<1> lws (256);


  if(alpha == 1.0 && beta == 1.0) {
    q.submit([&] (handler &h) {
        auto ycoefs = d_ycoefs.get_access<sycl_write>(h);
        auto xcoefs = d_xcoefs.get_access<sycl_read>(h);
        h.parallel_for<class dyx_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < n) ycoefs[i] += xcoefs[i];
            });
        });
  } else if (beta == 1.0) {
    q.submit([&] (handler &h) {
        auto ycoefs = d_ycoefs.get_access<sycl_write>(h);
        auto xcoefs = d_xcoefs.get_access<sycl_read>(h);
        h.parallel_for<class dyax_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < n) ycoefs[i] += alpha*xcoefs[i];
            });
        });
  } else if (alpha == 1.0) {
    q.submit([&] (handler &h) {
        auto ycoefs = d_ycoefs.get_access<sycl_read_write>(h);
        auto xcoefs = d_xcoefs.get_access<sycl_read>(h);
        h.parallel_for<class yxby_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < n) ycoefs[i] = xcoefs[i] + beta * ycoefs[i];
            });
        });
  } else if (beta == 0.0) {
    q.submit([&] (handler &h) {
        auto ycoefs = d_ycoefs.get_access<sycl_write>(h);
        auto xcoefs = d_xcoefs.get_access<sycl_read>(h);
        h.parallel_for<class yax_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < n) ycoefs[i] = alpha*xcoefs[i];
            });
        });
  } else {
    q.submit([&] (handler &h) {
        auto ycoefs = d_ycoefs.get_access<sycl_read_write>(h);
        auto xcoefs = d_xcoefs.get_access<sycl_read>(h);
        h.parallel_for<class yaxby_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < n) ycoefs[i] = alpha*xcoefs[i]+beta*ycoefs[i];
            });
        });
  }

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
    queue &q,
    buffer<typename Vector::ScalarType,1> &d_xcoefs,
    buffer<typename Vector::ScalarType,1> &d_ycoefs)
{
  const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

  //typedef typename Vector::ScalarType Scalar;
  //const Scalar*  xcoefs = &x.coefs[0];
  //const Scalar*  ycoefs = &y.coefs[0];

  MINIFE_SCALAR result = 0;


#ifdef ONEAPI_REDUCTION
  buffer<MINIFE_SCALAR, 1> d_result(&result, 1);
  d_result.set_final_data(nullptr);

  range<1> gws((n+255)/256*256);
  range<1> lws(256);

  // use Intel-specific reduction
  q.submit([&] (handler &h) {
    auto xcoefs_acc = d_xcoefs.template get_access<sycl_read>(h);
    auto ycoefs_acc = d_ycoefs.template get_access<sycl_read>(h);
    auto result_acc = d_result.template get_access<sycl_read_write>(h);
    h.parallel_for(nd_range<1>(gws, lws), 
      ext::oneapi::reduction(result_acc, result, std::plus<MINIFE_SCALAR>()),
      [=] (nd_item<1> item, auto& result_acc) {
      int i = item.get_global_id(0);
      if (i < n) result_acc += xcoefs_acc[i] * ycoefs_acc[i];
    });
  });

  q.submit([&] (handler &h) {
    auto result_acc = d_result.template get_access<sycl_read>(h);
    h.copy(result_acc, &result);
  });
#else
  // consistent with the reduction in the miniFE-cuda 
  int NWI = std::min(1024, (n+255)/256) * 256;
  range<1> gws (NWI);
  range<1> lws (256);
  buffer<MINIFE_SCALAR, 1> d_sop (1024); // sum-of-product

  q.submit([&] (handler &h) {
    auto d = d_sop.template get_access<sycl_discard_write>(h);
    h.fill(d, (MINIFE_SCALAR)0);
  });

  q.submit([&] (handler &h) {
    auto x = d_xcoefs.template get_access<sycl_read>(h);
    auto y = d_ycoefs.template get_access<sycl_read>(h);
    auto d = d_sop.template get_access<sycl_write>(h);
    accessor<MINIFE_SCALAR, 1, sycl_read_write, access::target::local> red(256, h);
    h.parallel_for<class xy_dot_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      MINIFE_SCALAR sum = 0;
      int lid = item.get_local_id(0);
      for(int idx=item.get_global_id(0);idx<n;idx+=item.get_group_range(0) * item.get_local_range(0)) {
        sum+=x[idx]*y[idx];
      }

      //Do a shared memory reduction on the dot product
      red[lid]=sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
        item.barrier(access::fence_space::local_space);
        if(lid<n)  {sum+=red[lid+n]; red[lid]=sum;}
      }

      //save partial dot products
      if(lid==0) d[item.get_group(0)]=sum;
    });
  });

  q.submit([&] (handler &h) {
    auto d = d_sop.template get_access<sycl_read_write>(h);
    accessor<MINIFE_SCALAR, 1, sycl_read_write, access::target::local> red(256, h);
    h.parallel_for<class final_reduce>(nd_range<1>(lws, lws), [=] (nd_item<1> item) {
      int lid = item.get_local_id(0);
      MINIFE_SCALAR sum = d[lid];
      red[lid]=sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
        item.barrier(access::fence_space::local_space);
        if(lid<n)  {sum+=red[lid+n]; red[lid]=sum;}
      }
      //save final dot product at the front
      if(lid==0) d[0]=sum;
    });
  });

  q.submit([&] (handler &h) {
    auto d = d_sop.template get_access<sycl_read>(h, range<1>(1));
    h.copy(d, &result);
  });
#endif

  q.wait();

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
dot_r2(const Vector& x, queue &q, buffer<typename Vector::ScalarType,1> &d_xcoefs)
{
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  std::cout << "[" << myrank << "] Starting dot..." << std::endl;
#endif
#endif

  const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

  //typedef typename Vector::ScalarType Scalar;

  MINIFE_SCALAR result = 0;

  //for(int i=0; i<n; ++i) {
  // result += xcoefs[i] * xcoefs[i];
  //}

#ifdef ONEAPI_REDUCTION
  range<1> gws ((n+255)/256*256);
  range<1> lws (256);

  buffer<MINIFE_SCALAR, 1> d_result(&result, 1);
  d_result.set_final_data(nullptr);

  // use Intel-specific reduction
  q.submit([&] (handler &h) {
    auto xcoefs_acc = d_xcoefs.template get_access<sycl_read>(h);
    auto result_acc = d_result.template get_access<sycl_read_write>(h);
    h.parallel_for<class reduction_kernel>(nd_range<1>(gws, lws), 
      ext::oneapi::reduction(result_acc, result, std::plus<MINIFE_SCALAR>()),
      [=] (nd_item<1> item, auto& result_acc) {
      int i = item.get_global_id(0);
      if (i < n) result_acc += xcoefs_acc[i] * xcoefs_acc[i];
    });
  });
  q.submit([&] (handler &h) {
    auto result_acc = d_result.template get_access<sycl_read>(h);
    h.copy(result_acc, &result);
  });
#else
  // consistent with the reduction in the miniFE-cuda 
  int NWI = std::min(1024, (n+255)/256) * 256;
  range<1> gws (NWI);
  range<1> lws (256);
  buffer<MINIFE_SCALAR, 1> d_sop (1024); // sum-of-product

  q.submit([&] (handler &h) {
    auto d = d_sop.template get_access<sycl_discard_write>(h);
    h.fill(d, (MINIFE_SCALAR)0);
  });

  q.submit([&] (handler &h) {
    auto x = d_xcoefs.template get_access<sycl_read>(h);
    auto d = d_sop.template get_access<sycl_write>(h);
    accessor<MINIFE_SCALAR, 1, sycl_read_write, access::target::local> red(256, h);
    h.parallel_for<class xx_dot_kernel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      MINIFE_SCALAR sum = 0;
      int lid = item.get_local_id(0);
      for(int idx=item.get_global_id(0);idx<n;idx+=item.get_group_range(0) * item.get_local_range(0)) {
        sum+=x[idx]*x[idx];
      }

      //Do a shared memory reduction on the dot product
      red[lid]=sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
        item.barrier(access::fence_space::local_space);
        if(lid<n)  {sum+=red[lid+n]; red[lid]=sum;}
      }

      //save partial dot products
      if(lid==0) d[item.get_group(0)]=sum;
    });
  });

  q.submit([&] (handler &h) {
    auto d = d_sop.template get_access<sycl_read_write>(h);
    accessor<MINIFE_SCALAR, 1, sycl_read_write, access::target::local> red(256, h);
    h.parallel_for<class final_reduce2>(nd_range<1>(lws, lws), [=] (nd_item<1> item) {
      int lid = item.get_local_id(0);
      MINIFE_SCALAR sum = d[lid];
      red[lid]=sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
        item.barrier(access::fence_space::local_space);
        if(lid<n)  {sum+=red[lid+n]; red[lid]=sum;}
      }
        //save final dot product at the front
      if(lid==0) d[0]=sum;
    });
  });

  q.submit([&] (handler &h) {
    auto d = d_sop.template get_access<sycl_read>(h, range<1>(1));
    h.copy(d, &result);
  });

#endif

  q.wait();

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

