#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <ctype.h>
#include "cu.h"
#include "base.h"
#include "prna.h"
#include "util.h"
#include "param.h"


/* penalty for a helix terminated by a pair containing a U */
DEV static real_t terminal_U_penalty(const base_t *s, const int i, const int j, param_t p)
{
  return s[i] == U || s[j] == U ? p->terminal_AU_penalty : RCONST(0.);
}

DEV static real_t dangle_3p_energy(const base_t *s,
				   const int i,
				   const int j,
				   const int ip1,
                                   param_t p)
{
  return p->dangle_3p[s[i]][s[j]][s[ip1]] + terminal_U_penalty(s,i,j,p);
}

DEV static real_t dangle_5p_energy(const base_t *s,
				   const int i,
				   const int j,
				   const int jm1,
                                   param_t p)
{
  return p->dangle_5p[s[i]][s[j]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}

DEV static real_t terminal_stack(const base_t *s,
		                 const int i,
				 const int j,
				 const int ip1,
				 const int jm1,
                                 param_t p)
{
  return p->tstack[s[i]][s[j]][s[ip1]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}

DEV static real_t terminal_stack_multibranch(const base_t *s,
					     const int i,
					     const int j,
					     const int ip1,
					     const int jm1,
					     param_t p)
{
  return p->tstackm[s[i]][s[j]][s[ip1]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}


DEV static const real_t *lookup_find(const base_t *s, const int d, param_t p)
{
  int i;
  switch (d) {
  case 3:
    for (i = 0; i < p->ntriloop; i++)
      if (sequences_match(s, p->triloop[i].seq, d+2))
	return &p->triloop[i].val;
    break;
  case 4:
    for (i = 0; i < p->ntloop; i++)
      if (sequences_match(s, p->tloop[i].seq, d+2))
	return &p->tloop[i].val;
    break;
  case 6:
    for (i = 0; i < p->nhexaloop; i++)
      if (sequences_match(s, p->hexaloop[i].seq, d+2))
	return &p->hexaloop[i].val;
    break;
  }
  return 0;
}

/***
 * Energy of a hairpin loop with d unpaired bases, d = j-i-1
 * s[i] is paired with s[j]
 * s[i+1] is mismatched with s[j-1]
 ***/
DEV static real_t hairpin_loop_energy(const base_t *s,
				      const int i,
				      const int j,
				      const int d,
                                      param_t p)
{
  /* Lookup tables for special hairpin loops */
  const real_t *val;
  if ((val = lookup_find(&s[i],d,p)))
    return *val;

  /* Hairpin loop initiation penalty */
  real_t e;
  if (d > LOOP_MAX)
    e = p->hairpin_loop_initiation[LOOP_MAX] + p->Extrapolation_for_large_loops *
      LOG((real_t) d / LOOP_MAX);
  else
    e = p->hairpin_loop_initiation[d];

  if (d == 3) {
    if (contains_only_base(C,d,&s[i+1]))
      e += p->c_hairpin_of_3;
    e += terminal_U_penalty(s,i,j,p);
  } else {
    e += p->tstackh[s[i]][s[j]][s[i+1]][s[j-1]];
    if (contains_only_base(C,d,&s[i+1]))
      e += p->c_hairpin_slope*d + p->c_hairpin_intercept;
  }

  if (s[i] == G && s[j] == U && i > 1 && s[i-1] == G && s[i-2] == G)
    e += p->bonus_for_GGG_hairpin;

  return e;
}

DEV static real_t real_min(real_t a, real_t b) { return a < b ? a : b; }

/***
 * Energy of an internal/bulge loop with d1, d2 unpaired bases,
 *   d1 = ip-i-1,  d2 = j-jp-1
 * s[i] is paired with s[j]
 * s[i+1] is mismatched sith s[j-1]
 * s[ip-1] is mismatched with s[jp+1]
 * s[ip] is paired with s[jp]
 ***/
DEV static real_t internal_loop_energy(const base_t *s,
				       const int i,
				       const int j,
				       const int ip,
				       const int jp,
				       const int d1,
				       const int d2,
                                       param_t p)
{
  /* Bulge loops */
  if (d1 == 0 || d2 == 0) {
    real_t e = p->bulge_loop_initiation[d1+d2];
    if (d1 == 1 || d2 == 1) { /* single-nucleotide bulge */
      e += p->stack[s[i]][s[j]][s[ip]][s[jp]];
      if ((d1 == 1 && s[i+1] == C && (s[i] == C || s[i+2] == C)) ||
          (d2 == 1 && s[j-1] == C && (s[j] == C || s[j-2] == C)))
        e += p->Bonus_for_Single_C_bulges_adjacent_to_C;
    } else {
      e += terminal_U_penalty(s,i,j,p);
      e += terminal_U_penalty(s,ip,jp,p);
    }
    return e;
  }

  /* Small internal loops */
  if (d1 == 1 && d2 == 1)
    return p->int11[s[i]][s[i+1]][s[i+2]][s[j-2]][s[j-1]][s[j]];
  if (d1 == 2 && d2 == 2)
    return p->int22[s[i]][s[ip]][s[j]][s[jp]][s[i+1]][s[i+2]][s[j-1]][s[j-2]];
  if (d1 == 1 && d2 == 2)
    return p->int21[s[i]][s[j]][s[i+1]][s[j-1]][s[jp+1]][s[ip]][s[jp]];
  if (d1 == 2 && d2 == 1)
    return p->int21[s[jp]][s[ip]][s[jp+1]][s[ip-1]][s[i+1]][s[j]][s[i]];

  /* Larger internal loops */
  tab4_t *sp;
  if (d1 == 1 || d2 == 1)
    sp = &p->tstacki1n;
  else if ((d1 == 2 && d2 == 3) || (d1 == 3 && d2 == 2))
    sp = &p->tstacki23;
  else
    sp = &p->tstacki;
  return p->internal_loop_initiation[d1+d2] +
    real_min(p->fm_array_first_element * ABS(d1-d2), p->maximum_correction) +
    (*sp)[s[i]][s[j]][s[i+1]][s[j-1]] +
    (*sp)[s[jp]][s[ip]][s[jp+1]][s[ip-1]];

}

/* return -ln(e^-a + e^-b) */
DEV static real_t free_energy_sum(const real_t a, const real_t b)
{
  if (a < b)
    return a - LOG1P(EXP(a-b));
  else if (b < a)
    return b - LOG1P(EXP(b-a));
  else
    return a - LOG(RCONST(2.));
}

DEV static void free_energy_accumulate(real_t *a, const real_t b)
{
  *a = free_energy_sum(*a,b);
}

DEV HOST static int int_min(int a, int b) { return a < b ? a : b; }

DEV HOST static int ind(int i, int j, int n)
{
  return i*n + j;
}

DEV HOST static int upper_triangle_index(int i, int j)
{
  return (j*(j-1))/2 + i;
}

DEV HOST inline static int can_pair(int i, int j, int n, const int *__restrict bcp)
{
    if (i>=0 && j<=n-1 && i != j && j>=0 && i<=n-1){
      if (i < j)
        return bcp[upper_triangle_index(i, j)];
      else
        return bcp[upper_triangle_index(j, i)];
    }
    else
      return 0;
}

DEV static int wrap(int i, int n)
{
  return i >= n ? i-n : i;
}

DEV static int is_exterior(int i, int j)
{
  return j < i;
}

DEV static int is_interior(int i, int j)
{
  return i < j;
}

DEV HOST real_t* array_val(real_t *__restrict a, int i, int j, int n, const int *__restrict bcp)
{
  return can_pair(i,j,n,bcp) ? &a[ind(i,j,n)] : 0;
}


#ifdef __CUDACC__
#include <sycl/sycl.hpp>
#define ISTART wi.get_group(2)
#define IINC wi.get_group_range(2)
#else
#define ISTART 0
#define IINC 1
#endif

GLOBAL static void calc_hairpin_stack_exterior_multibranch
(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
#endif
 const int d,
 const int n,
 const base_t *__restrict s,
 const int *__restrict bcp,
 real_t *__restrict v,
 const real_t *__restrict x,
 const real_t *__restrict w5,
 const real_t *__restrict w3,
 const param_t p)
{
  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) || !can_pair(i,j,n,bcp))
      continue;
    real_t vij = INF;
    if (i != n-1 && j != 0) {
      /* hairpin loop */
      if (is_interior(i,j))
        vij = hairpin_loop_energy(s,i,j,d,p);
      /* stack */
      if (can_pair(i+1,j-1,n,bcp) && !(is_interior(i,j) && d <= LOOP_MIN-2))
        free_energy_accumulate(&vij, p->stack[s[i]][s[j]][s[i+1]][s[j-1]] + v[ind(i+1,j-1,n)]);
    }
    /* exterior loop */
    if (is_exterior(i,j)) {
      free_energy_accumulate(&vij, w3[i+1] + w5[j-1] + terminal_U_penalty(s,i,j,p));
      if (i != n-1)
        free_energy_accumulate(&vij, w3[i+2] + w5[j-1] + dangle_3p_energy(s,i,j,i+1,p));
      if (j != 0)
        free_energy_accumulate(&vij, w3[i+1] + w5[j-2] + dangle_5p_energy(s,i,j,j-1,p));
      if (i != n-1 && j != 0)
        free_energy_accumulate(&vij, w3[i+2] + w5[j-2] + terminal_stack(s,i,j,i+1,j-1,p));
    }
    /* multibranch loop */
    if (d > 2*LOOP_MIN + 3 && i != n-1 && j != 0) {
      free_energy_accumulate(&vij, x[ind((d-2)%5,i+1,n)] + terminal_U_penalty(s,i,j,p) + p->a + p->c);
      if (i != n-2)
        free_energy_accumulate(&vij, x[ind((d-3)%5,i+2,n)] + dangle_3p_energy(s,i,j,i+1,p) + p->a + p->b + p->c);
      if (j != 1)
        free_energy_accumulate(&vij, x[ind((d-3)%5,i+1,n)] + dangle_5p_energy(s,i,j,j-1,p) + p->a + p->b + p->c);
      if (i != n-2 && j != 1)
        free_energy_accumulate(&vij, x[ind((d-4)%5,i+2,n)] + terminal_stack_multibranch(s,i,j,i+1,j-1,p) + p->a + 2*p->b + p->c);
    }
    v[ind(i,j,n)] = vij;
  }
}


#ifdef __CUDACC__

#define NTHREAD 128
#define THREAD_X 8
#define THREAD_Y 16

#if THREAD_X*THREAD_Y != NTHREAD
#error THREAD_X * THREAD_Y must be equal to NTHREAD
#endif

DEV static void free_energy_reduce(sycl::nd_item<3> wi, real_t *buf, real_t *x, int tid, int nt)
{
  buf[tid] = *x;
  for (nt /= 2, wi.barrier(sycl::access::fence_space::local_space); nt > 0;
       nt /= 2, wi.barrier(sycl::access::fence_space::local_space))
    if (tid < nt)
      free_energy_accumulate(&buf[tid], buf[tid+nt]);
  if (tid == 0)
    *x = buf[0];
}

#endif /* __CUDACC__ */

GLOBAL static void calc_internal
(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
 real_t *buf,
#endif
 const int d,
 const int n,
 const base_t *__restrict s,
 const int *__restrict bcp,
 real_t *__restrict v,
 const param_t p)
{
  int i;
#ifdef __CUDACC__
  const int tx = wi.get_local_id(2);
  const int ty = wi.get_local_id(1);
  const int dimx = wi.get_local_range(2);
  const int dimy = wi.get_local_range(1);
#endif

  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) ||
	(is_interior(i,j) && d <= LOOP_MIN+2) ||
	!can_pair(i,j,n,bcp))
      continue;
    real_t vij = INF;
#ifdef __CUDACC__
    const int d1start = tx;
    const int d1inc = dimx;
#else
    const int d1start = 0;
    const int d1inc = 1;
#endif
    const int dmax = int_min(LOOP_MAX, d-2);
    const int d1max = int_min(dmax, n-i-2);
    int d1;
    for (d1 = d1start; d1 <= d1max; d1 += d1inc) {
      const int ip = i+d1+1;
      const int d2max = int_min(dmax-d1, j-1);
#ifdef __CUDACC__
      const int d2start = d1 > 0 ? ty : ty + 1;
      const int d2inc = dimy;
#else
      const int d2start = d1 > 0 ? 0 : 1;
      const int d2inc = 1;
#endif
      int d2;
      for (d2 = d2start; d2 <= d2max; d2 += d2inc) {
        const int jp = j-d2-1;
        if (can_pair(ip,jp,n,bcp))
          free_energy_accumulate(&vij, internal_loop_energy(s,i,j,ip,jp,d1,d2,p)
				 + v[ind(ip,jp,n)]);
      }
    }
#ifdef __CUDACC__
    const int tid = tx * dimy + ty;
    free_energy_reduce(wi, buf, &vij, tid, dimx * dimy);
    if (tid != 0)
      continue;
#endif
    free_energy_accumulate(&v[ind(i,j,n)], vij);
  }
}

DEV static real_t coaxial_flush(const base_t *s,
				const int i,
				const int j,
				const int ip,
				const int jp,
                                param_t p)
{
  return terminal_U_penalty(s,i,j,p) + terminal_U_penalty(s,ip,jp,p) +
    p->coaxial[s[i]][s[j]][s[ip]][s[jp]];
}

DEV static real_t coaxial_mismatch1(const base_t *s,
				    const int i,
				    const int j,
				    const int ip,
				    const int jp,
                                    param_t p)
{
  return terminal_U_penalty(s,i,j,p) + terminal_U_penalty(s,ip,jp,p) +
    p->tstackcoax[s[j]][s[i]][s[j+1]][s[i-1]] +
    p->coaxstack[s[j+1]][s[i-1]][s[ip]][s[jp]];
}

DEV static real_t coaxial_mismatch2(const base_t *s,
				    const int i,
				    const int j,
				    const int ip,
				    const int jp,
                                    param_t p)
{
  return terminal_U_penalty(s,i,j,p) + terminal_U_penalty(s,ip,jp,p) +
    p->tstackcoax[s[jp]][s[ip]][s[jp+1]][s[ip-1]] +
    p->coaxstack[s[j]][s[i]][s[j+1]][s[jp+1]];
}

GLOBAL static void calc_coaxial
(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
 real_t *buf,
#endif
 const int d, /* diagonal - length of bases in between i and j, exclusive */
 const int n,
 const base_t *__restrict s,
 const int *__restrict bcp,
 real_t *__restrict v,
 const real_t *__restrict y,
 const real_t *__restrict w5,
 const real_t *__restrict w3,
 const param_t p)
{
  int i;
#ifdef __CUDACC__
  const int tx = wi.get_local_id(2);
  const int dimx = wi.get_local_range(2);
#endif

  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) || !can_pair(i,j,n,bcp))
      continue;
    const real_t *v1;
    real_t vij = INF;
    /* exterior */
    if (is_exterior(i,j)) {
      int k, kstart;
#ifdef __CUDACC__
      kstart = tx;
      const int kinc = dimx;
#else
      kstart = 0;
      const int kinc = 1;
#endif
      for (k = kstart; k < j - LOOP_MIN; k += kinc) {
	if ((v1 = array_val(v,k,j-1,n,bcp)))
	  free_energy_accumulate(&vij, w3[i+1] + w5[k-1] + coaxial_flush(s,k,j-1,j,i,p) + (*v1));
	if (j-2 >= 0) {
	  if (i < n-1 && (v1 = array_val(v,k,j-2,n,bcp)))
	    free_energy_accumulate(&vij, w3[i+2] + w5[k-1] + coaxial_mismatch2(s,k,j-2,j,i,p) + (*v1));
	  if ((v1 = array_val(v,k+1,j-2,n,bcp)))
	    free_energy_accumulate(&vij, w3[i+1] + w5[k-1] + coaxial_mismatch1(s,k+1,j-2,j,i,p) + (*v1));
	}
      }
#ifdef __CUDACC__
      kstart = i + LOOP_MIN + 1 + tx;
#else
      kstart = i + LOOP_MIN + 1;
#endif
      for (k = kstart; k < n; k += kinc) {
	if ((v1 = array_val(v,i+1,k,n,bcp)))
	  free_energy_accumulate(&vij, w3[k+1] + w5[j-1] + coaxial_flush(s,j,i,i+1,k,p) + (*v1));
	if (j > 0 && (v1 = array_val(v,i+2,k,n,bcp)))
	  free_energy_accumulate(&vij, w3[k+1] + w5[j-2] + coaxial_mismatch1(s,j,i,i+2,k,p) + (*v1));
	if ((v1 = array_val(v,i+2,k-1,n,bcp)))
	  free_energy_accumulate(&vij, w3[k+1] + w5[j-1] + coaxial_mismatch2(s,j,i,i+2,k-1,p) + (*v1));
      }
    } /* end exterior */

    /* multibranch */
    if (d > 2*LOOP_MIN + 3 && i != n-1 && j != 0) {
      int ktmp;
#ifdef __CUDACC__
      int ktmpstart = i + 2 + tx;
      const int ktmpinc = dimx;
#else
      int ktmpstart = i+2;
      const int ktmpinc = 1;
#endif
      for (ktmp = ktmpstart; ktmp < jtmp-2; ktmp += ktmpinc) {
	const int k = wrap(ktmp,n);
	if (k != n-1) {
	  if ((v1 = array_val(v,i+1,k,n,bcp)))
	    free_energy_accumulate(&vij, coaxial_flush(s,j,i,i+1,k,p) + (*v1) + p->a_2c +
				   y[ind(k+1,j-1,n)]);
	  if (ktmp+2 < jtmp-1 && i+1 != n-1 && k+1 != n-1 && (v1 = array_val(v,i+2,k,n,bcp))) {
	    const real_t tmp = (*v1) + p->a_2b_2c;
	    free_energy_accumulate(&vij, coaxial_mismatch2(s,j,i,i+2,k,p) + tmp + y[ind(k+2,j-1,n)]);
	    if (j != 1) {
	      free_energy_accumulate(&vij, coaxial_mismatch1(s,j,i,i+2,k,p) + tmp + y[ind(k+1,j-2,n)]);
	    }
	  }
	}
      }
#ifdef __CUDACC__
      ktmpstart = i+3 + tx;
#else
      ktmpstart = i+3;
#endif
      for (ktmp = ktmpstart; ktmp < jtmp-1; ktmp += ktmpinc) {
	const int k = wrap(ktmp,n);
	if (k != 0) {
	  if ((v1 = array_val(v,k,j-1,n,bcp)))
	    free_energy_accumulate(&vij, coaxial_flush(s,k,j-1,j,i,p) + (*v1) + p->a_2c +
				   y[ind(i+1,k-1,n)]);
	  if (j != 1 && ktmp > i+3 && (v1 = array_val(v,k,j-2,n,bcp))) {
	    const real_t tmp = (*v1) + p->a_2b_2c;
	    if (k != 1)
	      free_energy_accumulate(&vij, coaxial_mismatch1(s,k,j-2,j,i,p) + tmp + y[ind(i+1,k-2,n)]);
	    if (i != n-2)
	      free_energy_accumulate(&vij, coaxial_mismatch2(s,k,j-2,j,i,p) + tmp + y[ind(i+2,k-1,n)]);
	  }
	}
      }
    } /* end multibranch */
#ifdef __CUDACC__
    free_energy_reduce(wi, buf, &vij, tx, dimx);
    if (tx != 0)
      continue;
#endif
    free_energy_accumulate(&v[ind(i,j,n)], vij);
  } /* end loop over i */
} /* end calc_coaxial */

/***
 * For arrays w, wl, xl, two diagonals are stored.
 * Element i of the current diagonal - that is, w(i,j) -
 * is referenced as w[d%2][i].
 * Element i of the previous diagonal - that is, w(i,j-1) -
 * is referenced as w((d-1)%2,i)
 *
 * For array x, five diagonals are stored.
 * Similarly to w, x[ind(d%5,i,n)] refers to element i on
 * the current diagonal, and x[ind((d-k)%5,i,n)] to element i
 * on a previous diagonal d-k.
 * Specifically:
 *
 * x(i,j)       --> x(d%5,i,n)
 * x(i+1,j,n)   --> x((d-1)%5,i+1)
 * x(i+1,j-1,n) --> x((d-2)%5,i+1)
 * x(i+2,j-1,n) --> x((d-3)%5,i+1)
 * x(i+1,j-2,n) --> x((d-3)%5,i+1)
 * x(i+2,j-2,n) --> x((d-4)%5,i+1)
 ***/

GLOBAL static void calc_wl
(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
#endif
 const int d, /* diagonal - length of bases in between i and j, exclusive */
 const int n,
 const base_t *__restrict s,
 const int *__restrict bcp,
 real_t *__restrict v,
 real_t *__restrict z,
 real_t *__restrict wq,
 real_t *__restrict w,
 real_t *__restrict wl,
 const param_t p)
{
  int i;

  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if (is_exterior(i,j) && i-j <= LOOP_MIN)
      continue;
    real_t wqtmp = INF, wltmp = INF;
    const real_t *v1;
    if ((v1 = array_val(v,i,j,n,bcp))) {
      const real_t tmp = (*v1) + terminal_U_penalty(s,i,j,p);
      free_energy_accumulate(&wqtmp, tmp);
      free_energy_accumulate(&wltmp, tmp + p->c);
    }
    if (i != n-1 && (v1 = array_val(v,i+1,j,n,bcp))) {
      const real_t tmp = (*v1) + dangle_5p_energy(s,j,i+1,i,p);
      free_energy_accumulate(&wqtmp, tmp);
      free_energy_accumulate(&wltmp, tmp + p->b + p->c);
    }
    if (j != 0 && (v1 = array_val(v,i,j-1,n,bcp))) {
      const real_t tmp = (*v1) + dangle_3p_energy(s,j-1,i,j,p);
      free_energy_accumulate(&wqtmp, tmp);
      free_energy_accumulate(&wltmp, tmp + p->b + p->c);
    }
    if (i != n-1 && j != 0 && (v1 = array_val(v,i+1,j-1,n,bcp))) {
      const real_t tmp = (*v1) + terminal_stack_multibranch(s,j-1,i+1,j,i,p);
      free_energy_accumulate(&wqtmp, tmp);
      free_energy_accumulate(&wltmp, tmp + 2*p->b + p->c);
    }
    if (is_interior(i,j))
      wq[upper_triangle_index(i,j)] = wqtmp;
    /* WL array */
    wl[ind(d%2,i,n)] = z[ind(i,j,n)] = wltmp;
    if (i != n-1 && d > 0)
      free_energy_accumulate(&wl[ind(d%2,i,n)], wl[ind((d-1)%2,i+1,n)] + p->b);
    /* W array */
    w[ind(d%2,i,n)] = wl[ind(d%2,i,n)];
    if (j != 0 && d > 0)
      free_energy_accumulate(&w[ind(d%2,i,n)], w[ind((d-1)%2,i,n)] + p->b);
  } /* end loop over i */
} /* end calc_wl */

GLOBAL static void calc_xl
(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
 real_t *buf,
#endif
 const int d, /* diagonal - length of bases in between i and j, exclusive */
 const int n,
 const real_t *__restrict z,
 const real_t *__restrict yl,
 real_t *__restrict xl)
{
  int i;
#ifdef __CUDACC__
  const int tx = wi.get_local_id(2);
  const int dimx = wi.get_local_range(2);
#endif

  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if (is_exterior(i,j) && i-j <= LOOP_MIN)
      continue;
#ifdef __CUDACC__
    if (tx == 0)
      xl[ind(d%2,i,n)] = INF;
#else
    xl[ind(d%2,i,n)] = INF;
#endif
    if (is_interior(i,j) && d <= 2*LOOP_MIN+1)
      continue;
#ifdef __CUDACC__
    const int kstart = i+1 + tx;
    const int kinc = dimx;
#else
    const int kstart = i+1;
    const int kinc = 1;
#endif
    int ktmp;
    real_t tmp = INF;
    for (ktmp = kstart; ktmp < jtmp-1; ktmp += kinc) {
      if (ktmp != n-1) {
	const int k = wrap(ktmp,n);
	free_energy_accumulate(&tmp, z[ind(i,k,n)] + yl[ind(k+1,j,n)]);
      }
    }
#ifdef __CUDACC__
    free_energy_reduce(wi, buf, &tmp, tx, dimx);
    if (tx != 0)
      continue;
#endif
    free_energy_accumulate(&xl[ind(d%2,i,n)], tmp);
  } /* end loop over i */
} /* end calc_xl */

GLOBAL static void calc_z
(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
 real_t *buf,
#endif
 const int d, /* diagonal - length of bases in between i and j, exclusive */
 const int n,
 const base_t *__restrict s,
 const int *__restrict bcp,
 real_t *__restrict v,
 real_t *__restrict z,
 real_t *__restrict xl,
 real_t *__restrict wq,
 const param_t p)
{
  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) ||
	(is_interior(i,j) && d <= 2*LOOP_MIN+1))
      continue;
#ifdef __CUDACC__
    const int tx = wi.get_local_id(2);
    const int dimx = wi.get_local_range(2);
    const int kstart = i+LOOP_MIN+1+tx;
    const int kinc = dimx;
#else
    const int kstart = i+LOOP_MIN+1;
    const int kinc = 1;
#endif
    int ktmp;
    real_t tmp1 = INF, tmp2 = INF;
    for (ktmp = kstart; ktmp < jtmp-LOOP_MIN-1; ktmp += kinc) {
      const int k = wrap(ktmp,n);
      if (k == n-1)
	continue;
      real_t *v1, *v2;
      if ((v1 = array_val(v,i,k,n,bcp)) && (v2 = array_val(v,k+1,j,n,bcp)))
	free_energy_accumulate(&tmp1, (*v1) + (*v2) + coaxial_flush(s,i,k,k+1,j,p));
      if (j == 0 || k+1 == n-1)
	continue;
      if (i != n-1 && (v1 = array_val(v,i+1,k,n,bcp)) && (v2 = array_val(v,k+2,j,n,bcp)))
	free_energy_accumulate(&tmp2, (*v1) + (*v2) + coaxial_mismatch1(s,i+1,k,k+2,j,p));
      if ((v1 = array_val(v,i,k,n,bcp)) && (v2 = array_val(v,k+2,j-1,n,bcp)))
	free_energy_accumulate(&tmp2, (*v1) + (*v2) + coaxial_mismatch2(s,i,k,k+2,j-1,p));
    }
#ifdef __CUDACC__
    free_energy_reduce(wi, buf, &tmp1, tx, dimx);
    free_energy_reduce(wi, buf, &tmp2, tx, dimx);
    if (tx != 0)
      continue;
#endif
    if (is_interior(i,j))
      free_energy_accumulate(&wq[upper_triangle_index(i,j)], free_energy_sum(tmp1,tmp2));
    const real_t wcoax = free_energy_sum(tmp1 + 2*p->c, tmp2 + 2*p->b + 2*p->c);
    free_energy_accumulate(&z[ind(i,j,n)], wcoax);
    free_energy_accumulate(&xl[ind(d%2,i,n)], wcoax);
  } /* end loop over i */
} /* end calc_z */

GLOBAL static void calc_x
(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
#endif
 const int d, /* diagonal - length of bases in between i and j, exclusive */
 const int n,
 real_t *__restrict yl,
 real_t *__restrict y,
 const real_t *__restrict w,
 const real_t *__restrict wl,
 real_t *__restrict xl,
 real_t *__restrict x,
 const param_t p)
{
  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if (is_exterior(i,j) && i-j <= LOOP_MIN)
      continue;
    x[ind(d%5,i,n)] = INF;
    if (d > 2*LOOP_MIN+1 || is_exterior(i,j)) {
      if (i != n-1)
	free_energy_accumulate(&xl[ind(d%2,i,n)], xl[ind((d-1)%2,i+1,n)] + p->b);
      /* x array */
      x[ind(d%5,i,n)] = xl[ind(d%2,i,n)];
      if (j != 0)
	free_energy_accumulate(&x[ind(d%5,i,n)], x[ind((d-1)%5,i,n)] + p->b);
    }
    yl[ind(i,j,n)] = free_energy_sum(wl[ind(d%2,i,n)], xl[ind(d%2,i,n)]);
    y[ind(i,j,n)] = free_energy_sum(w[ind(d%2,i,n)], x[ind(d%5,i,n)]);
  } /* end loop over i */
} /* end calc_x */

GLOBAL static void init_w5_and_w3(int n, real_t *w5, real_t *w3)
{
  w5[-1] = w5[0] = w3[n-1] = w3[n] = 0;
}

GLOBAL static void calc_w5_and_w3(
#ifdef __CUDACC__
 sycl::nd_item<3> &wi,
 real_t *buf,
#endif
  const int d,
  const int n,
  real_t *__restrict w5,
  real_t *__restrict w3,
  const real_t *__restrict wq)
{
#ifdef __CUDACC__
  const int tx = wi.get_local_id(2);
  const int dimx = wi.get_local_range(2);
  const int istart = tx;
  const int iinc = dimx;
#else
  const int istart = 0;
  const int iinc = 1;
#endif
  real_t w5tmp = INF, w3tmp = INF;
  int i;
  for (i = istart; i + LOOP_MIN <= d; i += iinc) {
    free_energy_accumulate(&w5tmp, w5[i-1] + wq[upper_triangle_index(i,d+1)]);
    free_energy_accumulate(&w3tmp, w3[n-i] + wq[upper_triangle_index(n-d-2,n-i-1)]);
  }
#ifdef __CUDACC__
  free_energy_reduce(wi, buf, &w5tmp, tx, dimx);
  free_energy_reduce(wi, buf, &w3tmp, tx, dimx);
  if (tx != 0)
    return;
#endif
  w5[d+1] = w5[d];
  w3[n-d-2] = w3[n-d-1];
  free_energy_accumulate(&w5[d+1], w5tmp);
  free_energy_accumulate(&w3[n-d-2], w3tmp);
} /* end calc_w5_and_w3 */

prna_t prna_new(const char *s, param_t par, int quiet, int *base_cp)
{
  prna_t p = (prna_t) safe_malloc(sizeof(struct prna));
  memset(p, 0, sizeof(struct prna));

  const int n = p->n = strlen(s);
  printf("sequence length = %d\n", n);

  p->seq = (base_t *) safe_malloc(n*sizeof(base_t));
  p->base_can_pair = base_cp;
  sequence_from_string(p->seq, s);
  p->v = (real_t *) safe_malloc(n*n*sizeof(real_t));
  p->w5 = (real_t *) safe_malloc((n+1)*sizeof(real_t)) + 1;
  p->w3 = (real_t *) safe_malloc((n+1)*sizeof(real_t));

#ifdef __CUDACC__ /* do multithreaded fill on GPU */

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("Performing Calculation on GPU\n");
  real_t *d_v = sycl::malloc_device<real_t>(n*n, q);
  real_t *d_w5 = sycl::malloc_device<real_t>(n+1, q);
  d_w5++;
  real_t *d_w3 = sycl::malloc_device<real_t>(n+1, q);
  real_t *d_z = sycl::malloc_device<real_t>(n*n, q);
  real_t *d_yl = sycl::malloc_device<real_t>(n*n, q);
  real_t *d_y = sycl::malloc_device<real_t>(n*n, q);
  real_t *d_wq = sycl::malloc_device<real_t>(n*(n-1)/2, q);
  real_t *d_w = sycl::malloc_device<real_t>(n*2, q);
  real_t *d_wl = sycl::malloc_device<real_t>(n*2, q);
  real_t *d_xl = sycl::malloc_device<real_t>(n*2, q);
  real_t *d_x = sycl::malloc_device<real_t>(n*5, q);

  struct param *d_par = sycl::malloc_device<struct param>(1, q);
  q.memcpy(d_par, par, sizeof(struct param));

  base_t *d_s = sycl::malloc_device<base_t>(n, q);
  q.memcpy(d_s, p->seq, n*sizeof(base_t));

  int *d_bcp = sycl::malloc_device<int>(n*(n-1)/2, q);
  q.memcpy(d_bcp, p->base_can_pair, (n*(n-1)/2)*sizeof(int));

  q.submit([&] (sycl::handler &cgh) {
    cgh.single_task<class init_w5_w3>([=] () {
      init_w5_and_w3(n, d_w5, d_w3);
    });
  });

  sycl::range<3> lws1(1,1,1);
  sycl::range<3> gws1(1,1,n);
  sycl::range<3> lws2(1, THREAD_Y, THREAD_X);
  sycl::range<3> gws2(1, THREAD_Y, n*THREAD_X);
  sycl::range<3> lws3(1,1,NTHREAD);
  sycl::range<3> gws3(1,1,n*NTHREAD);
  sycl::range<3> lws4(1,1,1);
  sycl::range<3> gws4(1,1,n);
  sycl::range<3> lws5(1,1,NTHREAD);
  sycl::range<3> gws5(1,1,n*NTHREAD);
  sycl::range<3> lws6(1,1,NTHREAD);
  sycl::range<3> gws6(1,1,n*NTHREAD);
  sycl::range<3> lws7(1,1,1);
  sycl::range<3> gws7(1,1,n);
  sycl::range<3> lws8(1,1,NTHREAD);
  sycl::range<3> gws8(1,1,NTHREAD);

  for (int d = 0; d < n-1; d++) {

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k_calc_hairpin>(
        sycl::nd_range<3>(gws1, lws1), [=] (sycl::nd_item<3> wi) {
        calc_hairpin_stack_exterior_multibranch(wi, d, n, d_s,
          d_bcp, d_v, d_x, d_w5, d_w3, d_par);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<real_t, 1> lmem(sycl::range<1>(NTHREAD), cgh);
      cgh.parallel_for<class k_calc_internal>(
        sycl::nd_range<3>(gws2, lws2), [=] (sycl::nd_item<3> wi) {
        calc_internal(wi, lmem.get_pointer(), d, n, d_s, d_bcp, d_v, d_par);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<real_t, 1> lmem(sycl::range<1>(NTHREAD), cgh);
      cgh.parallel_for<class k_calc_coaxial>(
        sycl::nd_range<3>(gws3, lws3), [=] (sycl::nd_item<3> wi) {
        calc_coaxial(wi, lmem.get_pointer(), d, n,
                     d_s, d_bcp, d_v, d_y, d_w5, d_w3, d_par);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k_calc_wl>(
        sycl::nd_range<3>(gws4, lws4), [=] (sycl::nd_item<3> wi) {
        calc_wl(wi, d, n, d_s, d_bcp, d_v, d_z,
                d_wq, d_w, d_wl, d_par);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<real_t, 1> lmem(sycl::range<1>(NTHREAD), cgh);
      cgh.parallel_for<class k_calc_xl>(
        sycl::nd_range<3>(gws5, lws5), [=] (sycl::nd_item<3> wi) {
        calc_xl(wi, lmem.get_pointer(), d, n, d_z, d_yl, d_xl);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<real_t, 1> lmem(sycl::range<1>(NTHREAD), cgh);
      cgh.parallel_for<class k_calc_z>(
        sycl::nd_range<3>(gws6, lws6), [=] (sycl::nd_item<3> wi) {
        calc_z(wi, lmem.get_pointer(), d, n,
               d_s, d_bcp, d_v, d_z, d_xl, d_wq, d_par);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k_calc_x>(
        sycl::nd_range<3>(gws7, lws7), [=] (sycl::nd_item<3> wi) {
        calc_x(wi, d, n, d_yl, d_y, d_w, d_wl, d_xl, d_x, d_par);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<real_t, 1> lmem(sycl::range<1>(NTHREAD), cgh);
      cgh.parallel_for<class k_calc_w5_w3>(
        sycl::nd_range<3>(gws8, lws8), [=] (sycl::nd_item<3> wi) {
        calc_w5_and_w3(wi, lmem.get_pointer(), d, n, d_w5, d_w3, d_wq);
      });
    });
  }

  q.memcpy(p->v, d_v, n*n*sizeof(base_t));
  q.memcpy(p->w5 - 1, d_w5 - 1, (n+1)*sizeof(base_t));
  q.memcpy(p->w3, d_w3, (n+1)*sizeof(base_t));

  q.wait();

  sycl::free(d_v, q);
  sycl::free(d_w5 - 1, q);
  sycl::free(d_w3, q);
  sycl::free(d_z, q);
  sycl::free(d_yl, q);
  sycl::free(d_y, q);
  sycl::free(d_wq, q);
  sycl::free(d_w, q);
  sycl::free(d_wl, q);
  sycl::free(d_xl, q);
  sycl::free(d_x, q);
  sycl::free(d_par, q);
  sycl::free(d_s, q);
  sycl::free(d_bcp, q);
#else /* do serial fill on CPU */

#define ALLOC(a,sz) a = (real_t *) safe_malloc((sz)*sizeof(real_t))

  printf("Performing Calculations on CPU\n");

  real_t *z, *yl, *y, *wq, *w, *wl, *xl, *x;
  ALLOC(z,n*n);
  ALLOC(yl,n*n);
  ALLOC(y,n*n);
  ALLOC(wq,n*(n-1)/2);
  ALLOC(w,2*n);
  ALLOC(wl,2*n);
  ALLOC(xl,2*n);
  ALLOC(x,5*n);

  init_w5_and_w3(n,p->w5,p->w3);

  int d;
  for (d = 0; d < n-1; d++) {
    calc_hairpin_stack_exterior_multibranch(d, n, p->seq, p->base_can_pair, p->v, x, p->w5, p->w3, par);
    calc_internal(d, n, p->seq, p->base_can_pair, p->v, par);
    calc_coaxial(d, n, p->seq, p->base_can_pair, p->v, y, p->w5, p->w3, par);
    calc_wl(d, n, p->seq, p->base_can_pair, p->v, z, wq, w, wl, par);
    calc_xl(d, n, z, yl, xl);
    calc_z(d, n, p->seq, p->base_can_pair, p->v, z, xl, wq, par);
    calc_x(d, n, yl, y, w, wl, xl, x, par);
    calc_w5_and_w3(d, n, p->w5, p->w3, wq);
  }

  free(z);
  free(yl);
  free(y);
  free(wq);
  free(w);
  free(wl);
  free(xl);
  free(x);
#endif /* __CUDACC__ */

  return p;


} /* end prna_new */

void prna_delete(prna_t p)
{
  if (p) {
    if (p->seq)
      free(p->seq);
    if (p->v)
      free(p->v);
    if (p->w5 - 1)
      free(p->w5 - 1);
    if (p->w3)
      free(p->w3);
    free(p);
  }
}

#define SHOWARR(a)	 \
  if (p->a) {		      \
    int i, j;	      \
    for (i = 0; i < n; i++) { \
      printf("%s%4d: ",#a,i+1);				\
      for (j = 0; j < n; j++) {				\
	const real_t *aij = array_val(p->a,i,j,n,bcp);	\
	printf(RF" ", aij ? (*aij)*RT37 : INF);	\
      }								\
      printf("\n");						\
    }								\
  }

#define SHOW(a)							\
  if (p->a) {								\
    int i;								\
    printf("%s: ",#a);							\
    for (i = 0; i < n; i++)						\
      printf(RF" ", p->a[i] * RT37);				\
    printf("\n");							\
  }									\

void prna_show(const prna_t p)
{
  int i, n = p->n;
  const base_t *s = p->seq;
  const int *bcp = p->base_can_pair;
  printf("n: %d\n", n);
  printf("seq: ");
  for (i = 0; i < n; i++)
    printf("%c", base_as_char(s[i]));
  printf("\n");
  SHOWARR(v);
  SHOW(w5);
  SHOW(w3);
}

static real_t free_energy_of_pair(const prna_t p, int i, int j)
{
  const int n = p->n;
  //const base_t *s = p->seq;
  const int *bcp = p->base_can_pair;
  if (can_pair(i,j,n,bcp)){
    return *array_val(p->v,i,j,n,bcp) + *array_val(p->v,j,i,n,bcp) - p->w3[0];
  }
  else
    return INF;
}

real_t probability_of_pair(const prna_t p, int i, int j)
{
  return exp(-free_energy_of_pair(p,i,j));
}

real_t get_v_array(const prna_t p, int i, int j)
{
  const int n = p->n;
  const int *bcp = p->base_can_pair;

  if (can_pair(i,j,n,bcp)){
    return *array_val(p->v,i,j,n,bcp);
  }
  else
    return -INF;
}

real_t get_w3_array(const prna_t p, int i)
{
  return p->w3[i];
}

real_t get_w5_array(const prna_t p, int i)
{
  return p->w5[i];
}

void prna_write_neg_log10_probabilities(const prna_t p, const char *fn)
{
  FILE *f = safe_fopen(fn,"w");
  int i, j;
  fprintf(f,"%d\n%-8s%-8s-log10(probability)\n",p->n,"i","j");
  for (i = 0; i < p->n; i++)
    for (j = i+1; j < p->n; j++)
      if (can_pair(i,j,p->n,p->base_can_pair))
	fprintf(f,"%-8d%-8d" RF "\n", i+1, j+1,
	        free_energy_of_pair(p,i,j)/MATHFN(log)(10));
  fclose(f);
}

void prna_write_probability_matrix(const prna_t p, const char *fn)
{
  FILE *f = safe_fopen(fn,"w");
  const int n = p->n;
  //const base_t *s = p->seq;
  const int *bcp = p->base_can_pair;
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      fprintf(f,RF" ",
	      can_pair(i,j,n,bcp) ? probability_of_pair(p,i,j) : 0);
    fprintf(f,"\n");
  }
  fclose(f);
}

static void write_ct_structure(FILE *f, const char *s, int n, const int *pair)
{
  char fmt[256];
  sprintf(fmt,"%d",n);
  int ns = strlen(fmt)+1;
  if (ns < 5)
    ns = 5;
  sprintf(fmt,"%%%dd",ns);
  int i;
  for (i = 0; i < n; i++) {
    fprintf(f,fmt,i+1);
    fprintf(f,"%2c   ",s[i]);
    fprintf(f,fmt,i);
    fprintf(f,fmt,i == n-1 ? 0 : i+2);
    fprintf(f,fmt,pair[i] == i ? 0 : pair[i]+1);
    fprintf(f,fmt,i+1);
    fprintf(f,"\n");
  }
}

static void unpair(int *pair, int i)
{
  const int j = pair[i];
  pair[i] = i;
  pair[j] = j;
}

static int is_paired(const int *pair, int i)
{
  return pair[i] != i;
}

static void remove_helices_shorter_than(int min_helix_length, int *pair, int n)
{
  int i;
  for (i = 0; i < n-2; i++) {
    int j = pair[i];
    if (j <= i)
      continue;
    int npair = 1;
    while (pair[i+1] == j-1 || pair[i+2] == j-1 || pair[i+1] == j-2) {
      if (pair[i+1] == j-1)
	;
      else if (pair[i+2] == j-1) {
	if (is_paired(pair,i+1))
	  unpair(pair,i+1);
	i++;
      } else
	j--;
      i++;
      j--;
      npair++;
    }
    if (npair < min_helix_length) {
      unpair(pair,i);
      if (i >= 2) {
	while (pair[i-1] == j+1 || pair[i-2] == j+1 || pair[i-1] == j+2) {
	  if (pair[i-1] == j+1)
	    unpair(pair,i-1);
	  else if (pair[i-2] == j+1) {
	    unpair(pair,i-2);
	    i--;
	  } else {
	    unpair(pair,i-1);
	    j++;
	  }
	  i--;
	  j++;
	}
      } else if (i == 1) {
	while (pair[i-1] == j+1 || pair[i-1] == j+2) {
	  if (pair[i-1] == j+1)
	    unpair(pair,i-1);
	  else {
	    unpair(pair,i-1);
	    j++;
	  }
	  i--;
	  j++;
	}
      }
    }
  }
} /* end remove_helices_shorter_than */

void prna_write_probknot(const prna_t p, const char *fn, const char *s, int min_helix_length)
{
  const int n = p->n;
  int *pair = (int *) safe_malloc(n*sizeof(int));
  int i;
  for (i = 0; i < n; i++) {
    pair[i] = i; /* unpaired */
    int j;
    for (j = 0; j < n; j++)
      if (free_energy_of_pair(p,i,j) < free_energy_of_pair(p,i,pair[i]))
	      pair[i] = j;
  }
  for (i = 0; i < n; i++)
    if (pair[pair[i]] != i)
      pair[i] = i; /* unpaired */
  if (min_helix_length > 1)
    remove_helices_shorter_than(min_helix_length,pair,n);
  /* write the structure */
  if (fn) {
    FILE *f = safe_fopen(fn,"w");
    write_ct_structure(f,s,n,pair);
    fclose(f);
  } else {
    write_ct_structure(stdout,s,n,pair);
  }
  free(pair);
}

int *generate_bcp(const char *s)
{
  int length = strlen(s);
  int i, j;

  int *base_cp = (int *) safe_malloc((length*(length-1)/2)*sizeof(int));

  base_t *seq = (base_t *) safe_malloc(length*sizeof(base_t));
  sequence_from_string(seq, s);

  for (i=0; i<length; i++){
    for (j=i+1; j<length; j++){
      if ((j-i < LOOP_MIN+1) || !isupper(s[i]) || !isupper(s[j])){
        base_cp[(j*(j-1))/2 + i]=0;
      }
      else{
          base_cp[upper_triangle_index(i,j)]=is_canonical_pair(seq[i],seq[j]) && ((i > 0 && j < length-1 && is_canonical_pair(seq[i-1],seq[j+1]))
            || (j-i>=LOOP_MIN+3 && is_canonical_pair(seq[i+1],seq[j-1])));
      }
    }
  }

  return base_cp;
}
