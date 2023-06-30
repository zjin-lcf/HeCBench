#ifndef _cg_solve_hpp_
#define _cg_solve_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia	Corporation
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

#include <cmath>
#include <limits>

#include <Vector_functions.hpp>
#include <mytimer.hpp>

#include <outstream.hpp>
#include <sycl/sycl.hpp>


namespace miniFE {

template<typename Scalar>
void print_vec(const std::vector<Scalar>& vec, const std::string& name)
{
  for(size_t i=0; i<vec.size(); ++i) {
    std::cout << name << "["<<i<<"]: " << vec[i] << std::endl;
  }
}

template<typename VectorType>
bool breakdown(typename VectorType::ScalarType inner,
               const VectorType& v,
               const VectorType& w,
      sycl::queue &q,
      typename VectorType::ScalarType *d_v,
      typename VectorType::ScalarType *d_w)
{
  typedef typename VectorType::ScalarType Scalar;
  typedef typename TypeTraits<Scalar>::magnitude_type magnitude;

//This is code that was copied from Aztec, and originally written
//by my hero, Ray Tuminaro.
//
//Assuming that inner = <v,w> (inner product of v and w),
//v and w are considered orthogonal if
//  |inner| < 100 * ||v||_2 * ||w||_2 * epsilon

  magnitude vnorm = std::sqrt(dot(v,v, q, d_v, d_v));
  magnitude wnorm = std::sqrt(dot(w,w, q, d_w, d_w));
  return std::abs(inner) <= 100*vnorm*wnorm*std::numeric_limits<magnitude>::epsilon();
}

template<typename OperatorType,
         typename VectorType,
         typename Matvec>
void
cg_solve(OperatorType& A,
         const VectorType& b,
         VectorType& x,
         Matvec matvec,
         typename OperatorType::LocalOrdinalType max_iter,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance,
         typename OperatorType::LocalOrdinalType& num_iters,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr,
         timer_type* my_cg_times)
{
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
  typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
  typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  timer_type t0 = 0, tWAXPY = 0, tDOT = 0, tMATVEC = 0, tMATVECDOT = 0;
  timer_type total_time = mytimer();

  int myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  if (!A.has_local_indices) {
    std::cerr << "miniFE::cg_solve ERROR, A.has_local_indices is false, needs to be true. This probably means "
       << "miniFE::make_local_matrix(A) was not called prior to calling miniFE::cg_solve."
       << std::endl;
    return;
  }

  size_t nrows = A.rows.size();
  LocalOrdinalType ncols = A.num_cols;

  VectorType r(b.startIndex, nrows);
  VectorType p(0, ncols);
  VectorType Ap(b.startIndex, nrows);

  normr = 0;
  magnitude_type rtrans = 0;
  magnitude_type oldrtrans = 0;

  LocalOrdinalType print_freq = max_iter/10;
  if (print_freq>50) print_freq = 50;
  if (print_freq<1)  print_freq = 1;

  ScalarType one = 1.0;
  ScalarType zero = 0.0;

  MINIFE_SCALAR* MINIFE_RESTRICT r_ptr = &r.coefs[0];
  MINIFE_SCALAR* MINIFE_RESTRICT p_ptr = &p.coefs[0];
  MINIFE_SCALAR* MINIFE_RESTRICT Ap_ptr = &Ap.coefs[0];
  MINIFE_SCALAR* MINIFE_RESTRICT x_ptr = &x.coefs[0];
  const MINIFE_SCALAR* MINIFE_RESTRICT b_ptr = &b.coefs[0];

  const MINIFE_LOCAL_ORDINAL* MINIFE_RESTRICT const Arowoffsets = &A.row_offsets[0];
  const MINIFE_GLOBAL_ORDINAL* MINIFE_RESTRICT const Acols	= &A.packed_cols[0];
  const MINIFE_SCALAR* MINIFE_RESTRICT const Acoefs             = &A.packed_coefs[0];

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  MINIFE_SCALAR *d_r = sycl::malloc_device<MINIFE_SCALAR>(r.coefs.size(), q);
  q.memcpy(d_r, r_ptr, sizeof(MINIFE_SCALAR) * r.coefs.size());

  MINIFE_SCALAR *d_p = sycl::malloc_device<MINIFE_SCALAR>(p.coefs.size(), q);
  q.memcpy(d_p, p_ptr, sizeof(MINIFE_SCALAR) * p.coefs.size());

  MINIFE_SCALAR *d_Ap = sycl::malloc_device<MINIFE_SCALAR>(Ap.coefs.size(), q);
  q.memcpy(d_Ap, Ap_ptr, sizeof(MINIFE_SCALAR) * Ap.coefs.size());

  MINIFE_SCALAR *d_x = sycl::malloc_device<MINIFE_SCALAR>(x.coefs.size(), q);
  q.memcpy(d_x, x_ptr, sizeof(MINIFE_SCALAR) * x.coefs.size());

  MINIFE_SCALAR *d_b = sycl::malloc_device<MINIFE_SCALAR>(b.coefs.size(), q);
  q.memcpy(d_b, b_ptr, sizeof(MINIFE_SCALAR) * b.coefs.size());

  LocalOrdinalType *d_Arowoffsets = sycl::malloc_device<LocalOrdinalType>(A.row_offsets.size(), q);
  q.memcpy(d_Arowoffsets, Arowoffsets, sizeof(LocalOrdinalType) * A.row_offsets.size());

  GlobalOrdinalType *d_Acols = sycl::malloc_device<GlobalOrdinalType>(A.packed_cols.size(), q);
  q.memcpy(d_Acols, Acols, sizeof(GlobalOrdinalType) * A.packed_cols.size());

  MINIFE_SCALAR *d_Acoefs = sycl::malloc_device<MINIFE_SCALAR>(A.packed_coefs.size(), q);
  q.memcpy(d_Acoefs, Acoefs, sizeof(MINIFE_SCALAR) * A.packed_coefs.size());

  TICK(); waxpby(one, x, zero, x, p, q, d_x, d_x, d_p); TOCK(tWAXPY);

#ifdef MINIFE_DEBUG
  print_vec(p.coefs, "p");
#endif

  TICK();
  matvec(A, p, Ap, q, d_Arowoffsets, d_Acols, d_Acoefs, d_p, d_Ap);
  TOCK(tMATVEC);

  TICK();
  waxpby(one, b, -one, Ap, r, q, d_b, d_Ap, d_r);
  TOCK(tWAXPY);

  TICK(); rtrans = dot_r2(r, q, d_r); TOCK(tDOT);

#ifdef MINIFE_DEBUG
std::cout << "rtrans="<<rtrans<<std::endl;
#endif

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "brkdown_tol = " << brkdown_tol << std::endl;

  std::cout << "Starting CG Solve Phase..." << std::endl;
#endif

  for(LocalOrdinalType k=1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      TICK();
      waxpby(one, r, zero, r, p, q, d_r, d_r, d_p);
      TOCK(tWAXPY);
    }
    else {
      oldrtrans = rtrans;
      TICK(); rtrans = dot_r2(r, q, d_r); TOCK(tDOT);
      magnitude_type beta = rtrans/oldrtrans;
      TICK();
      daxpby(one, r, beta, p, q, d_r, d_p);
      TOCK(tWAXPY);
    }

    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    TICK();
    matvec(A, p, Ap, q, d_Arowoffsets, d_Acols, d_Acoefs, d_p, d_Ap);
    TOCK(tMATVEC);

    TICK(); p_ap_dot = dot(Ap, p, q, d_Ap, d_p); TOCK(tDOT);

#ifdef MINIFE_DEBUG
    os << "iter " << k << ", p_ap_dot = " << p_ap_dot;
    os.flush();
#endif
    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 || breakdown(p_ap_dot, Ap, p, q, d_Ap, d_p)) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
#ifdef MINIFE_DEBUG
        os << "ERROR, numerical breakdown!"<<std::endl;
#endif
        //update the timers before jumping out.
        my_cg_times[WAXPY] = tWAXPY;
        my_cg_times[DOT] = tDOT;
        my_cg_times[MATVEC] = tMATVEC;
        my_cg_times[TOTAL] = mytimer() - total_time;
        return;
      }
      else brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans/p_ap_dot;
#ifdef MINIFE_DEBUG
    os << ", rtrans = " << rtrans << ", alpha = " << alpha << std::endl;
#endif

    TICK();
    daxpby(alpha, p, one, x, q, d_p, d_x);
    daxpby(-alpha, Ap, one, r, q, d_Ap, d_r);
    TOCK(tWAXPY);

    num_iters = k;
  }

  q.memcpy(x_ptr, d_x, sizeof(MINIFE_SCALAR) * x.coefs.size()).wait();

  my_cg_times[WAXPY] = tWAXPY;
  my_cg_times[DOT] = tDOT;
  my_cg_times[MATVEC] = tMATVEC;
  my_cg_times[MATVECDOT] = tMATVECDOT;
  my_cg_times[TOTAL] = mytimer() - total_time;

  sycl::free(d_r, q);
  sycl::free(d_p, q);
  sycl::free(d_Ap, q);
  sycl::free(d_x, q);
  sycl::free(d_b, q);
  sycl::free(d_Arowoffsets, q);
  sycl::free(d_Acols, q);
  sycl::free(d_Acoefs, q);
}

}//namespace miniFE

#endif

