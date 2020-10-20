/** GPU solver for 2D lid-driven cavity problem, using finite difference method
 * \file main_gpu.cu
 *
 * \author Kyle E. Niemeyer
 * \date 09/27/2012
 *
 * Solve the incompressible, isothermal 2D Navier–Stokes equations for a square
 * lid-driven cavity on a GPU (via CUDA), using the finite difference method.
 * To change the grid resolution, modify "NUM". In addition, the problem is controlled
 * by the Reynolds number ("Re_num").
 * 
 * Based on the methodology given in Chapter 3 of "Numerical Simulation in Fluid
 * Dynamics", by M. Griebel, T. Dornseifer, and T. Neunhoeffer. SIAM, Philadelphia,
 * PA, 1998.
 * 
 * Boundary conditions:
 * u = 0 and v = 0 at x = 0, x = L, y = 0
 * u = ustar at y = H
 * v = 0 at y = H
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

// CUDA libraries

/** Problem size along one side; total number of cells is this squared */
#define NUM 512

// block size
#define BLOCK_SIZE 128

/** Double precision */
#define DOUBLE

#ifdef DOUBLE
#define Real double

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

#define SMALL 1.0e-10;

/** Reynolds number */
const Real Re_num = 1000.0;

/** SOR relaxation parameter */
const Real omega = 1.7;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9;

/** Safety factor for time step modification */
const Real tau = 0.5;

/** Body forces in x- and y- directions */
const Real gx = 0.0;
const Real gy = 0.0;

/** Domain size (non-dimensional) */
#define xLength 1.0
#define yLength 1.0
#else
#define Real float

// replace double functions with float versions
#undef fmin
#define fmin fminf
#undef fmax
#define fmax fmaxf
#undef fabs
#define fabs fabsf
#undef sqrt
#define sqrt sqrtf

#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define SMALL 1.0e-10f;

/** Reynolds number */
const Real Re_num = 1000.0f;

/** SOR relaxation parameter */
const Real omega = 1.7f;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9f;

/** Safety factor for time step modification */
const Real tau = 0.5f;

/** Body forces in x- and y- directions */
const Real gx = 0.0f;
const Real gy = 0.0f;

/** Domain size (non-dimensional) */
#define xLength 1.0f
#define yLength 1.0f
#endif


/** Mesh sizes */
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

/** Max macro (type safe, from GNU) */
//#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

/** Min macro (type safe) */
//#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

// map two-dimensional indices to one-dimensional memory
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

///////////////////////////////////////////////////////////////////////////////
  
void set_BCs_host (Real* u, Real* v) 
{
  int ind;

  // loop through rows and columns
  for (ind = 0; ind < NUM + 2; ++ind) {

    // left boundary
    u(0, ind) = ZERO;
    v(0, ind) = -v(1, ind);

    // right boundary
    u(NUM, ind) = ZERO;
    v(NUM + 1, ind) = -v(NUM, ind);

    // bottom boundary
    u(ind, 0) = -u(ind, 1);
    v(ind, 0) = ZERO;

    // top boundary
    u(ind, NUM + 1) = TWO - u(ind, NUM);
    v(ind, NUM) = ZERO;

    if (ind == NUM) {
      // left boundary
      u(0, 0) = ZERO;
      v(0, 0) = -v(1, 0);
      u(0, NUM + 1) = ZERO;
      v(0, NUM + 1) = -v(1, NUM + 1);

      // right boundary
      u(NUM, 0) = ZERO;
      v(NUM + 1, 0) = -v(NUM, 0);
      u(NUM, NUM + 1) = ZERO;
      v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

      // bottom boundary
      u(0, 0) = -u(0, 1);
      v(0, 0) = ZERO;
      u(NUM + 1, 0) = -u(NUM + 1, 1);
      v(NUM + 1, 0) = ZERO;

      // top boundary
      u(0, NUM + 1) = TWO - u(0, NUM);
      v(0, NUM) = ZERO;
      u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
      v(ind, NUM + 1) = ZERO;
    } // end if

  } // end for

} // end set_BCs_host

///////////////////////////////////////////////////////////////////////////////
  
void set_BCs (Real* u, Real* v, sycl::nd_item<3> item_ct1) 
{
  int ind = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;

  // left boundary
  u(0, ind) = ZERO;
  v(0, ind) = -v(1, ind);

  // right boundary
  u(NUM, ind) = ZERO;
  v(NUM + 1, ind) = -v(NUM, ind);

  // bottom boundary
  u(ind, 0) = -u(ind, 1);
  v(ind, 0) = ZERO;

  // top boundary
  u(ind, NUM + 1) = TWO - u(ind, NUM);
  v(ind, NUM) = ZERO;

  if (ind == NUM) {
    // left boundary
    u(0, 0) = ZERO;
    v(0, 0) = -v(1, 0);
    u(0, NUM + 1) = ZERO;
    v(0, NUM + 1) = -v(1, NUM + 1);

    // right boundary
    u(NUM, 0) = ZERO;
    v(NUM + 1, 0) = -v(NUM, 0);
    u(NUM, NUM + 1) = ZERO;
    v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

    // bottom boundary
    u(0, 0) = -u(0, 1);
    v(0, 0) = ZERO;
    u(NUM + 1, 0) = -u(NUM + 1, 1);
    v(NUM + 1, 0) = ZERO;

    // top boundary
    u(0, NUM + 1) = TWO - u(0, NUM);
    v(0, NUM) = ZERO;
    u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
    v(ind, NUM + 1) = ZERO;
  } // end if

} // end set_BCs

///////////////////////////////////////////////////////////////////////////////

  
void calculate_F (const Real dt, const Real* u, const Real* v,
    Real* F, sycl::nd_item<3> item_ct1) 
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  if (col == NUM) {
    // right boundary, F_ij = u_ij
    // also do left boundary
    F(0, row) = u(0, row);
    F(NUM, row) = u(NUM, row);
  } else {

    // u velocities
    Real u_ij = u(col, row);
    Real u_ip1j = u(col + 1, row);
    Real u_ijp1 = u(col, row + 1);
    Real u_im1j = u(col - 1, row);
    Real u_ijm1 = u(col, row - 1);

    // v velocities
    Real v_ij = v(col, row);
    Real v_ip1j = v(col + 1, row);
    Real v_ijm1 = v(col, row - 1);
    Real v_ip1jm1 = v(col + 1, row - 1);

    // finite differences
    Real du2dx, duvdy, d2udx2, d2udy2;

    du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) -
              (u_im1j + u_ij) * (u_im1j + u_ij)) +
             mix_param * (sycl::fabs(u_ij + u_ip1j) * (u_ij - u_ip1j) -
                          sycl::fabs(u_im1j + u_ij) * (u_im1j - u_ij))) /
            (FOUR * dx);
    duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) -
             (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij) +
             mix_param * (sycl::fabs(v_ij + v_ip1j) * (u_ij - u_ijp1) -
                          sycl::fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) /
            (FOUR * dy);
    d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) / (dx * dx);
    d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) / (dy * dy);

    F(col, row) = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);

  } // end if

} // end calculate_F

///////////////////////////////////////////////////////////////////////////////

  
void calculate_G (const Real dt, const Real* u, const Real* v,
    Real* G, sycl::nd_item<3> item_ct1) 
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  if (row == NUM) {
    // top and bottom boundaries
    G(col, 0) = v(col, 0);
    G(col, NUM) = v(col, NUM);

  } else {

    // u velocities
    Real u_ij = u(col, row);
    Real u_ijp1 = u(col, row + 1);
    Real u_im1j = u(col - 1, row);
    Real u_im1jp1 = u(col - 1, row + 1);

    // v velocities
    Real v_ij = v(col, row);
    Real v_ijp1 = v(col, row + 1);
    Real v_ip1j = v(col + 1, row);
    Real v_ijm1 = v(col, row - 1);
    Real v_im1j = v(col - 1, row);

    // finite differences
    Real dv2dy, duvdx, d2vdx2, d2vdy2;

    dv2dy =
        ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij) +
         mix_param * (sycl::fabs(v_ij + v_ijp1) * (v_ij - v_ijp1) -
                      sycl::fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) /
        (FOUR * dy);
    duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) -
             (u_im1j + u_im1jp1) * (v_im1j + v_ij) +
             mix_param * (sycl::fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) -
                          sycl::fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) /
            (FOUR * dx);
    d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (dx * dx);
    d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (dy * dy);

    G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);

  } // end if

} // end calculate_G

///////////////////////////////////////////////////////////////////////////////

  
void sum_pressure (const Real* pres_red, const Real* pres_black, 
    Real* pres_sum, sycl::nd_item<3> item_ct1, Real *sum_cache) 
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  // shared memory for block's sum

  int NUM_2 = NUM >> 1;

  Real pres_r = pres_red(col, row);
  Real pres_b = pres_black(col, row);

  // add squared pressure
  sum_cache[item_ct1.get_local_id(2)] = (pres_r * pres_r) + (pres_b * pres_b);

  // synchronize threads in block to ensure all thread values stored
  item_ct1.barrier();

  // add up values for block
  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (item_ct1.get_local_id(2) < i) {
      sum_cache[item_ct1.get_local_id(2)] +=
          sum_cache[item_ct1.get_local_id(2) + i];
    }
    item_ct1.barrier();
    i >>= 1;
  }

  // store block's summed values
  if (item_ct1.get_local_id(2) == 0) {
    pres_sum[item_ct1.get_group(1) + (item_ct1.get_group_range(1) *
                                      item_ct1.get_group(2))] = sum_cache[0];
  }

} // end sum_pressure

///////////////////////////////////////////////////////////////////////////////
  
void set_horz_pres_BCs (Real* pres_red, Real* pres_black,
                        sycl::nd_item<3> item_ct1) 
{
  int col = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  col = (col * 2) - 1;

  int NUM_2 = NUM >> 1;

  // p_i,0 = p_i,1
  pres_black(col, 0) = pres_red(col, 1);
  pres_red(col + 1, 0) = pres_black(col + 1, 1);

  // p_i,jmax+1 = p_i,jmax
  pres_red(col, NUM_2 + 1) = pres_black(col, NUM_2);
  pres_black(col + 1, NUM_2 + 1) = pres_red(col + 1, NUM_2);

} // end set_horz_pres_BCs

//////////////////////////////////////////////////////////////////////////////

  
void set_vert_pres_BCs (Real* pres_red, Real* pres_black,
                        sycl::nd_item<3> item_ct1) 
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;

  int NUM_2 = NUM >> 1;

  // p_0,j = p_1,j
  pres_black(0, row) = pres_red(1, row);
  pres_red(0, row) = pres_black(1, row);

  // p_imax+1,j = p_imax,j
  pres_black(NUM + 1, row) = pres_red(NUM, row);
  pres_red(NUM + 1, row) = pres_black(NUM, row);

} // end set_pressure_BCs

///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for red cells
 * 
 * \param[in]    dt      time-step size
 * \param[in]    F      array of discretized x-momentum eqn terms
 * \param[in]    G      array of discretized y-momentum eqn terms
 * \param[in]    pres_black  pressure values of black cells
 * \param[inout]  pres_red  pressure values of red cells
 */
  
void red_kernel (const Real dt, const Real* F, 
    const Real* G, const Real* pres_black,
    Real* pres_red, sycl::nd_item<3> item_ct1) 
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  int NUM_2 = NUM >> 1;      

  Real p_ij = pres_red(col, row);

  Real p_im1j = pres_black(col - 1, row);
  Real p_ip1j = pres_black(col + 1, row);
  Real p_ijm1 = pres_black(col, row - (col & 1));
  Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

  // right-hand side
  Real rhs = (((F(col, (2 * row) - (col & 1))
          - F(col - 1, (2 * row) - (col & 1))) / dx)
      + ((G(col, (2 * row) - (col & 1))
          - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

  pres_red(col, row) = p_ij * (ONE - omega) + omega * 
    (((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
     rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));

} // end red_kernel

///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for black cells
 * 
 * \param[in]    dt      time-step size
 * \param[in]    F      array of discretized x-momentum eqn terms
 * \param[in]    G      array of discretized y-momentum eqn terms
 * \param[in]    pres_red  pressure values of red cells
 * \param[inout]  pres_black  pressure values of black cells
 */
  
void black_kernel (const Real dt, const Real* F, 
    const Real* G, const Real* pres_red, 
    Real* pres_black, sycl::nd_item<3> item_ct1) 
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  int NUM_2 = NUM >> 1;

  Real p_ij = pres_black(col, row);

  Real p_im1j = pres_red(col - 1, row);
  Real p_ip1j = pres_red(col + 1, row);
  Real p_ijm1 = pres_red(col, row - ((col + 1) & 1));
  Real p_ijp1 = pres_red(col, row + (col & 1));

  // right-hand side
  Real rhs = (((F(col, (2 * row) - ((col + 1) & 1))
          - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
      + ((G(col, (2 * row) - ((col + 1) & 1))
          - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

  pres_black(col, row) = p_ij * (ONE - omega) + omega * 
    (((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
     rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));

} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

  
void calc_residual (const Real dt, const Real* F, const Real* G, 
    const Real* pres_red, const Real* pres_black,
    Real* res_array, sycl::nd_item<3> item_ct1, Real *sum_cache)
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  int NUM_2 = NUM >> 1;

  Real p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs, res, res2;

  // red point
  p_ij = pres_red(col, row);

  p_im1j = pres_black(col - 1, row);
  p_ip1j = pres_black(col + 1, row);
  p_ijm1 = pres_black(col, row - (col & 1));
  p_ijp1 = pres_black(col, row + ((col + 1) & 1));

  rhs = (((F(col, (2 * row) - (col & 1)) - F(col - 1, (2 * row) - (col & 1))) / dx)
      +  ((G(col, (2 * row) - (col & 1)) - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

  // calculate residual
  res = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
    + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

  // black point
  p_ij = pres_black(col, row);

  p_im1j = pres_red(col - 1, row);
  p_ip1j = pres_red(col + 1, row);
  p_ijm1 = pres_red(col, row - ((col + 1) & 1));
  p_ijp1 = pres_red(col, row + (col & 1));

  // right-hand side
  rhs = (((F(col, (2 * row) - ((col + 1) & 1)) - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
      +  ((G(col, (2 * row) - ((col + 1) & 1)) - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

  // calculate residual
  res2 = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
    + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

  // shared memory for block's sum

  sum_cache[item_ct1.get_local_id(2)] = (res * res) + (res2 * res2);

  // synchronize threads in block to ensure all residuals stored
  item_ct1.barrier();

  // add up squared residuals for block
  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (item_ct1.get_local_id(2) < i) {
      sum_cache[item_ct1.get_local_id(2)] +=
          sum_cache[item_ct1.get_local_id(2) + i];
    }
    item_ct1.barrier();
    i >>= 1;
  }

  // store block's summed residuals
  if (item_ct1.get_local_id(2) == 0) {
    res_array[item_ct1.get_group(1) + (item_ct1.get_group_range(1) *
                                       item_ct1.get_group(2))] = sum_cache[0];
  }
} 

///////////////////////////////////////////////////////////////////////////////

  
void calculate_u (const Real dt, const Real* F, 
    const Real* pres_red, const Real* pres_black, 
    Real* u, Real* max_u, sycl::nd_item<3> item_ct1, Real *max_cache)
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  // allocate shared memory to store max velocities

  max_cache[item_ct1.get_local_id(2)] = ZERO;

  int NUM_2 = NUM >> 1;
  Real new_u = ZERO;

  if (col != NUM) {

    Real p_ij, p_ip1j, new_u2;

    // red point
    p_ij = pres_red(col, row);
    p_ip1j = pres_black(col + 1, row);

    new_u = F(col, (2 * row) - (col & 1)) - (dt * (p_ip1j - p_ij) / dx);
    u(col, (2 * row) - (col & 1)) = new_u;

    // black point
    p_ij = pres_black(col, row);
    p_ip1j = pres_red(col + 1, row);

    new_u2 = F(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ip1j - p_ij) / dx);
    u(col, (2 * row) - ((col + 1) & 1)) = new_u2;

    // check for max of these two
    new_u = sycl::fmax(sycl::fabs(new_u), sycl::fabs(new_u2));

    if ((2 * row) == NUM) {
      // also test for max velocity at vertical boundary
      new_u = sycl::fmax(new_u, sycl::fabs( u(col, NUM + 1) ));
    }
  } else {
    // check for maximum velocity in boundary cells also
    new_u = sycl::fmax(sycl::fabs( u(NUM, (2 * row)) ), sycl::fabs( u(0, (2 * row)) ));
    new_u = sycl::fmax(sycl::fabs( u(NUM, (2 * row) - 1) ), new_u);
    new_u = sycl::fmax(sycl::fabs( u(0, (2 * row) - 1) ), new_u);
    new_u = sycl::fmax(sycl::fabs( u(NUM + 1, (2 * row)) ), new_u);
    new_u = sycl::fmax(sycl::fabs( u(NUM + 1, (2 * row) - 1) ), new_u);

  } // end if

  // store maximum u for block from each thread
  max_cache[item_ct1.get_local_id(2)] = new_u;

  // synchronize threads in block to ensure all velocities stored
  item_ct1.barrier();

  // calculate maximum for block
  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (item_ct1.get_local_id(2) < i) {
      max_cache[item_ct1.get_local_id(2)] =
          sycl::fmax(max_cache[item_ct1.get_local_id(2)],
                     max_cache[item_ct1.get_local_id(2) + i]);
    }
    item_ct1.barrier();
    i >>= 1;
  }

  // store block's maximum
  if (item_ct1.get_local_id(2) == 0) {
    max_u[item_ct1.get_group(1) +
          (item_ct1.get_group_range(1) * item_ct1.get_group(2))] = max_cache[0];
  }


} // end calculate_u

///////////////////////////////////////////////////////////////////////////////

  
void calculate_v (const Real dt, const Real* G, 
    const Real* pres_red, const Real* pres_black, 
    Real* v, Real* max_v, sycl::nd_item<3> item_ct1, Real *max_cache)
{
  int row = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2) + 1;
  int col = (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
            item_ct1.get_local_id(1) + 1;

  // allocate shared memory to store maximum velocities

  max_cache[item_ct1.get_local_id(2)] = ZERO;

  int NUM_2 = NUM >> 1;
  Real new_v = ZERO;

  if (row != NUM_2) {
    Real p_ij, p_ijp1, new_v2;

    // red pressure point
    p_ij = pres_red(col, row);
    p_ijp1 = pres_black(col, row + ((col + 1) & 1));

    new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
    v(col, (2 * row) - (col & 1)) = new_v;


    // black pressure point
    p_ij = pres_black(col, row);
    p_ijp1 = pres_red(col, row + (col & 1));

    new_v2 = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
    v(col, (2 * row) - ((col + 1) & 1)) = new_v2;


    // check for max of these two
    new_v = sycl::fmax(sycl::fabs(new_v), sycl::fabs(new_v2));

    if (col == NUM) {
      // also test for max velocity at vertical boundary
      new_v = sycl::fmax(new_v, sycl::fabs( v(NUM + 1, (2 * row)) ));
    }

  } else {

    if ((col & 1) == 1) {
      // black point is on boundary, only calculate red point below it
      Real p_ij = pres_red(col, row);
      Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

      new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
      v(col, (2 * row) - (col & 1)) = new_v;

    } else {
      // red point is on boundary, only calculate black point below it
      Real p_ij = pres_black(col, row);
      Real p_ijp1 = pres_red(col, row + (col & 1));

      new_v = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
      v(col, (2 * row) - ((col + 1) & 1)) = new_v;
    }

    // get maximum v velocity
    new_v = sycl::fabs(new_v);

    // check for maximum velocity in boundary cells also
    new_v = sycl::fmax(sycl::fabs( v(col, NUM) ), new_v);
    new_v = sycl::fmax(sycl::fabs( v(col, 0) ), new_v);

    new_v = sycl::fmax(sycl::fabs( v(col, NUM + 1) ), new_v);

  } // end if

  // store absolute value of velocity
  max_cache[item_ct1.get_local_id(2)] = new_v;

  // synchronize threads in block to ensure all velocities stored
  item_ct1.barrier();

  // calculate maximum for block
  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (item_ct1.get_local_id(2) < i) {
      max_cache[item_ct1.get_local_id(2)] =
          sycl::fmax(max_cache[item_ct1.get_local_id(2)],
                     max_cache[item_ct1.get_local_id(2) + i]);
    }
    item_ct1.barrier();
    i >>= 1;
  }

  // store block's summed residuals
  if (item_ct1.get_local_id(2) == 0) {
    max_v[item_ct1.get_group(1) +
          (item_ct1.get_group_range(1) * item_ct1.get_group(2))] = max_cache[0];
  }

} // end calculate_v

///////////////////////////////////////////////////////////////////////////////

int main (int argc, char *argv[])
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  // iterations for Red-Black Gauss-Seidel with SOR
  int iter = 0;
  const int it_max = 1000000;

  // SOR iteration tolerance
  const Real tol = 0.001;

  // time range
  const Real time_start = 0.0;
  const Real time_end = 0.001; //20.0;

  // initial time step size
  Real dt = 0.02;

  int size = (NUM + 2) * (NUM + 2);
  int size_pres = ((NUM / 2) + 2) * (NUM + 2);

  // arrays for pressure and velocity
  Real* F;
  Real* u;
  Real* G;
  Real* v;

  F = (Real *) calloc (size, sizeof(Real));
  u = (Real *) calloc (size, sizeof(Real));
  G = (Real *) calloc (size, sizeof(Real));
  v = (Real *) calloc (size, sizeof(Real));

  for (int i = 0; i < size; ++i) {
    F[i] = ZERO;
    u[i] = ZERO;
    G[i] = ZERO;
    v[i] = ZERO;
  }

  // arrays for pressure
  Real* pres_red;
  Real* pres_black;

  pres_red = (Real *) calloc (size_pres, sizeof(Real));
  pres_black = (Real *) calloc (size_pres, sizeof(Real));

  for (int i = 0; i < size_pres; ++i) {
    pres_red[i] = ZERO;
    pres_black[i] = ZERO;
  }

  // print problem size
  printf("Problem size: %d x %d \n", NUM, NUM);

  ////////////////////////////////////////
  // block and grid dimensions

  // boundary conditions kernel
  sycl::range<3> block_bcs(BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_bcs(NUM / BLOCK_SIZE, 1, 1);

  // pressure kernel
  sycl::range<3> block_pr(BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_pr(NUM / (2 * BLOCK_SIZE), NUM, 1);

  // block and grid dimensions for F
  sycl::range<3> block_F(BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_F(NUM / BLOCK_SIZE, NUM, 1);

  // block and grid dimensions for G
  sycl::range<3> block_G(BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_G(NUM / BLOCK_SIZE, NUM, 1);

  // horizontal pressure boundary conditions
  sycl::range<3> block_hpbc(BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_hpbc(NUM / (2 * BLOCK_SIZE), 1, 1);

  // vertical pressure boundary conditions
  sycl::range<3> block_vpbc(BLOCK_SIZE, 1, 1);
  sycl::range<3> grid_vpbc(NUM / (2 * BLOCK_SIZE), 1, 1);
  ///////////////////////////////////////////

  // residual variable
  Real* res;

  int size_res = grid_pr[0] * grid_pr[1];
  res = (Real *) calloc (size_res, sizeof(Real));

  // variables to store maximum velocities
  Real* max_u_arr;
  Real* max_v_arr;
  int size_max = grid_pr[0] * grid_pr[1];

  max_u_arr = (Real *) calloc (size_max, sizeof(Real));
  max_v_arr = (Real *) calloc (size_max, sizeof(Real));

  // pressure sum
  Real* pres_sum;
  pres_sum = (Real *) calloc (size_res, sizeof(Real));

  // set initial BCs
  set_BCs_host (u, v);

  Real max_u = SMALL;
  Real max_v = SMALL;
  // get max velocity for initial values (including BCs)
#pragma unroll
  for (int col = 0; col < NUM + 2; ++col) {
#pragma unroll
    for (int row = 1; row < NUM + 2; ++row) {
      max_u = fmax(max_u, fabs( u(col, row) ));
    }
  }

#pragma unroll
  for (int col = 1; col < NUM + 2; ++col) {
#pragma unroll
    for (int row = 0; row < NUM + 2; ++row) {
      max_v = fmax(max_v, fabs( v(col, row) ));
    }
  }

  ////////////////////////////////////////
  // allocate and transfer device memory
  Real* u_d;
  Real* F_d;
  Real* v_d;
  Real* G_d;

  Real* pres_red_d;
  Real* pres_black_d;
  Real* pres_sum_d;
  Real* res_d;

  Real* max_u_d;
  Real* max_v_d;

  u_d = sycl::malloc_device<double>(size, q_ct1);
  F_d = sycl::malloc_device<double>(size, q_ct1);
  v_d = sycl::malloc_device<double>(size, q_ct1);
  G_d = sycl::malloc_device<double>(size, q_ct1);

  pres_red_d = sycl::malloc_device<double>(size_pres, q_ct1);
  pres_black_d = sycl::malloc_device<double>(size_pres, q_ct1);

  pres_sum_d = sycl::malloc_device<double>(size_res, q_ct1);
  res_d = sycl::malloc_device<double>(size_res, q_ct1);
  max_u_d = sycl::malloc_device<double>(size_max, q_ct1);
  max_v_d = sycl::malloc_device<double>(size_max, q_ct1);

  // copy to device memory
  q_ct1.memcpy(u_d, u, size * sizeof(Real)).wait();
  q_ct1.memcpy(F_d, F, size * sizeof(Real)).wait();
  q_ct1.memcpy(v_d, v, size * sizeof(Real)).wait();
  q_ct1.memcpy(G_d, G, size * sizeof(Real)).wait();
  q_ct1.memcpy(pres_red_d, pres_red, size_pres * sizeof(Real)).wait();
  q_ct1.memcpy(pres_black_d, pres_black, size_pres * sizeof(Real)).wait();
  ////////////////////////////////////////

  Real time = time_start;

  // time-step size based on grid and Reynolds number
  Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

  // time iteration loop
  while (time < time_end) {

    // calculate time step based on stability and CFL
    dt = fmin((dx / max_u), (dy / max_v));
    dt = tau * fmin(dt_Re, dt);

    if ((time + dt) >= time_end) {
      dt = time_end - time;
    }

    // calculate F and G
    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = grid_F * block_F;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(block_F.get(2), block_F.get(1), block_F.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            calculate_F(dt, u_d, v_d, F_d, item_ct1);
          });
    });
    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = grid_G * block_G;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(block_G.get(2), block_G.get(1), block_G.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            calculate_G(dt, u_d, v_d, G_d, item_ct1);
          });
    });

    // get L2 norm of initial pressure
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<Real, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          sum_cache_acc_ct1(sycl::range<1>(128 /*BLOCK_SIZE*/), cgh);

      auto dpct_global_range = grid_pr * block_pr;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(block_pr.get(2), block_pr.get(1),
                                           block_pr.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            sum_pressure(pres_red_d, pres_black_d, pres_sum_d, item_ct1,
                         sum_cache_acc_ct1.get_pointer());
          });
    });
    q_ct1.memcpy(pres_sum, pres_sum_d, size_res * sizeof(Real)).wait();

    Real p0_norm = ZERO;
#pragma unroll
    for (int i = 0; i < size_res; ++i) {
      p0_norm += pres_sum[i];
    }

    p0_norm = sqrt(p0_norm / ((Real)(NUM * NUM)));
    if (p0_norm < 0.0001) {
      p0_norm = 1.0;
    }

    // ensure all kernels are finished
    //cudaDeviceSynchronize();

    Real norm_L2;

    // calculate new pressure
    // red-black Gauss-Seidel with SOR iteration loop
    for (iter = 1; iter <= it_max; ++iter) {

      // set pressure boundary conditions
      q_ct1.submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid_hpbc * block_hpbc;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(block_hpbc.get(2),
                                             block_hpbc.get(1),
                                             block_hpbc.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              set_horz_pres_BCs(pres_red_d, pres_black_d, item_ct1);
            });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid_vpbc * block_hpbc;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(block_hpbc.get(2),
                                             block_hpbc.get(1),
                                             block_hpbc.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              set_vert_pres_BCs(pres_red_d, pres_black_d, item_ct1);
            });
      });

      // ensure kernel finished
      //cudaDeviceSynchronize();

      // update red cells
      q_ct1.submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid_pr * block_pr;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(block_pr.get(2), block_pr.get(1),
                                             block_pr.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              red_kernel(dt, F_d, G_d, pres_black_d, pres_red_d, item_ct1);
            });
      });

      // ensure red kernel finished
      //cudaDeviceSynchronize();

      // update black cells
      q_ct1.submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid_pr * block_pr;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(block_pr.get(2), block_pr.get(1),
                                             block_pr.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              black_kernel(dt, F_d, G_d, pres_red_d, pres_black_d, item_ct1);
            });
      });

      // ensure red kernel finished
      //cudaDeviceSynchronize();

      // calculate residual values
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<Real, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sum_cache_acc_ct1(sycl::range<1>(128 /*BLOCK_SIZE*/), cgh);

        auto dpct_global_range = grid_pr * block_pr;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(block_pr.get(2), block_pr.get(1),
                                             block_pr.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              calc_residual(dt, F_d, G_d, pres_red_d, pres_black_d, res_d,
                            item_ct1, sum_cache_acc_ct1.get_pointer());
            });
      });

      // transfer residual value(s) back to CPU
      q_ct1.memcpy(res, res_d, size_res * sizeof(Real)).wait();

      norm_L2 = ZERO;
#pragma unroll
      for (int i = 0; i < size_res; ++i) {
        norm_L2 += res[i];
      }

      // calculate residual
      norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;

      // if tolerance has been reached, end SOR iterations
      if (norm_L2 < tol) {
        break;
      }  
    } // end for

    printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

    // calculate new velocities and transfer maximums back

    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<Real, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          max_cache_acc_ct1(sycl::range<1>(128 /*BLOCK_SIZE*/), cgh);

      auto dpct_global_range = grid_pr * block_pr;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(block_pr.get(2), block_pr.get(1),
                                           block_pr.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            calculate_u(dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d,
                        item_ct1, max_cache_acc_ct1.get_pointer());
          });
    });
    q_ct1.memcpy(max_u_arr, max_u_d, size_max * sizeof(Real)).wait();

    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<Real, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          max_cache_acc_ct1(sycl::range<1>(128 /*BLOCK_SIZE*/), cgh);

      auto dpct_global_range = grid_pr * block_pr;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(block_pr.get(2), block_pr.get(1),
                                           block_pr.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            calculate_v(dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d,
                        item_ct1, max_cache_acc_ct1.get_pointer());
          });
    });
    q_ct1.memcpy(max_v_arr, max_v_d, size_max * sizeof(Real)).wait();

    // get maximum u- and v- velocities
    max_v = SMALL;
    max_u = SMALL;

#pragma unroll
    for (int i = 0; i < size_max; ++i) {
      Real test_u = max_u_arr[i];
      max_u = fmax(max_u, test_u);

      Real test_v = max_v_arr[i];
      max_v = fmax(max_v, test_v);
    }

    // set velocity boundary conditions
    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = grid_bcs * block_bcs;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(block_bcs.get(2), block_bcs.get(1),
                                           block_bcs.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            set_BCs(u_d, v_d, item_ct1);
          });
    });

    dev_ct1.queues_wait_and_throw();

    // increase time
    time += dt;

    // single time step
    //break;

  } // end while

  // transfer final temperature values back
  q_ct1.memcpy(u, u_d, size * sizeof(Real)).wait();
  q_ct1.memcpy(v, v_d, size * sizeof(Real)).wait();
  q_ct1.memcpy(pres_red, pres_red_d, size_pres * sizeof(Real)).wait();
  q_ct1.memcpy(pres_black, pres_black_d, size_pres * sizeof(Real)).wait();

  // write data to file
  FILE * pfile;
  pfile = fopen("velocity_gpu.dat", "w");
  fprintf(pfile, "#x\ty\tu\tv\n");
  if (pfile != NULL) {
    for (int row = 0; row < NUM; ++row) {
      for (int col = 0; col < NUM; ++col) {

        Real u_ij = u[(col * NUM) + row];
        Real u_im1j;
        if (col == 0) {
          u_im1j = 0.0;
        } else {
          u_im1j = u[(col - 1) * NUM + row];
        }

        u_ij = (u_ij + u_im1j) / 2.0;

        Real v_ij = v[(col * NUM) + row];
        Real v_ijm1;
        if (row == 0) {
          v_ijm1 = 0.0;
        } else {
          v_ijm1 = v[(col * NUM) + row - 1];
        }

        v_ij = (v_ij + v_ijm1) / 2.0;

        fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
      }
    }
  }

  fclose(pfile);

  // free device memory
  sycl::free(u_d, q_ct1);
  sycl::free(v_d, q_ct1);
  sycl::free(F_d, q_ct1);
  sycl::free(G_d, q_ct1);
  sycl::free(pres_red_d, q_ct1);
  sycl::free(pres_black_d, q_ct1);
  sycl::free(max_u_d, q_ct1);
  sycl::free(max_v_d, q_ct1);
  sycl::free(pres_sum_d, q_ct1);
  sycl::free(res_d, q_ct1);

  free(pres_red);
  free(pres_black);
  free(u);
  free(v);
  free(F);
  free(G);
  free(max_u_arr);
  free(max_v_arr);
  free(res);
  free(pres_sum);

  return 0;
}
