// map two-dimensional indices to one-dimensional indices for device memory
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

void set_BCs (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    Real*__restrict__ u,
    Real*__restrict__ v) 
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int ind = item.get_global_id(2) + 1;

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
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);

} // end set_BCs

///////////////////////////////////////////////////////////////////////////////

void calculate_F (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real dt,
    const Real*__restrict__ u,
    const Real*__restrict__ v,
          Real*__restrict__ F) 
{  
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;

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

        du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
            + mix_param * (sycl::fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
              - sycl::fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (FOUR * dx);
        duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
            + mix_param * (sycl::fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
              - sycl::fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (FOUR * dy);
        d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) / (dx * dx);
        d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) / (dy * dy);

        F(col, row) = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);

      } // end if
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);

} // end calculate_F

///////////////////////////////////////////////////////////////////////////////

void calculate_G (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real dt,
    const Real*__restrict__ u,
    const Real*__restrict__ v,
          Real*__restrict__ G) 
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;

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

        dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
            + mix_param * (sycl::fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
              - sycl::fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (FOUR * dy);
        duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
            + mix_param * (sycl::fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
              - sycl::fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (FOUR * dx);
        d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (dx * dx);
        d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (dy * dy);

        G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);

      } // end if
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);

} // end calculate_G

///////////////////////////////////////////////////////////////////////////////

void sum_pressure (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real*__restrict__ pres_red,
    const Real*__restrict__ pres_black, 
          Real*__restrict__ pres_sum) 
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<Real, 1> sum_cache (sycl::range<1>(BLOCK_SIZE), cgh);

    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;
      int lid = item.get_local_id(2);

      int NUM_2 = NUM >> 1;

      Real pres_r = pres_red(col, row);
      Real pres_b = pres_black(col, row);

      // add squared pressure
      sum_cache[lid] = (pres_r * pres_r) + (pres_b * pres_b);

      // synchronize threads in block to ensure all thread values stored
      item.barrier(sycl::access::fence_space::local_space);

      // add up values for block
      int i = BLOCK_SIZE >> 1;
      while (i != 0) {
        if (lid < i) {
          sum_cache[lid] += sum_cache[lid + i];
        }
        item.barrier(sycl::access::fence_space::local_space);
        i >>= 1;
      }

      // store block's summed values
      if (lid == 0) {
        pres_sum[item.get_group_linear_id()] = sum_cache[0];
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);

} // end sum_pressure

///////////////////////////////////////////////////////////////////////////////
void set_horz_pres_BCs (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    Real*__restrict__ pres_red,
    Real*__restrict__ pres_black) 
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int col = item.get_global_id(2) + 1;
      col = (col * 2) - 1;

      int NUM_2 = NUM >> 1;

      // p_i,0 = p_i,1
      pres_black(col, 0) = pres_red(col, 1);
      pres_red(col + 1, 0) = pres_black(col + 1, 1);

      // p_i,jmax+1 = p_i,jmax
      pres_red(col, NUM_2 + 1) = pres_black(col, NUM_2);
      pres_black(col + 1, NUM_2 + 1) = pres_red(col + 1, NUM_2);
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);

} // end set_horz_pres_BCs

//////////////////////////////////////////////////////////////////////////////

void set_vert_pres_BCs (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    Real*__restrict__ pres_red,
    Real*__restrict__ pres_black) 
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;

      int NUM_2 = NUM >> 1;

      // p_0,j = p_1,j
      pres_black(0, row) = pres_red(1, row);
      pres_red(0, row) = pres_black(1, row);

      // p_imax+1,j = p_imax,j
      pres_black(NUM + 1, row) = pres_red(NUM, row);
      pres_red(NUM + 1, row) = pres_black(NUM, row);
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);

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
void red_kernel (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real dt,
    const Real*__restrict__ F, 
    const Real*__restrict__ G,
    const Real*__restrict__ pres_black,
          Real*__restrict__ pres_red) 
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;

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
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);

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
void black_kernel (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real dt,
    const Real*__restrict__ F, 
    const Real*__restrict__ G,
    const Real*__restrict__ pres_red, 
          Real*__restrict__ pres_black) 
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;

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
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

void calc_residual (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real dt,
    const Real*__restrict__ F,
    const Real*__restrict__ G, 
    const Real*__restrict__ pres_red,
    const Real*__restrict__ pres_black,
          Real*__restrict__ res_array)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<Real, 1> sum_cache (sycl::range<1>(BLOCK_SIZE), cgh);

    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;
      int lid = item.get_local_id(2);

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

      sum_cache[lid] = (res * res) + (res2 * res2);

      // synchronize threads in block to ensure all residuals stored
      item.barrier(sycl::access::fence_space::local_space);

      // add up squared residuals for block
      int i = BLOCK_SIZE >> 1;
      while (i != 0) {
        if (lid < i) {
          sum_cache[lid] += sum_cache[lid + i];
        }
        item.barrier(sycl::access::fence_space::local_space);
        i >>= 1;
      }

      // store block's summed residuals
      if (lid == 0) {
        res_array[item.get_group_linear_id()] = sum_cache[0];
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
} 

///////////////////////////////////////////////////////////////////////////////

void calculate_u (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real dt,
    const Real*__restrict__ F, 
    const Real*__restrict__ pres_red,
    const Real*__restrict__ pres_black, 
          Real*__restrict__ u,
          Real*__restrict__ max_u)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<Real, 1> max_cache (sycl::range<1>(BLOCK_SIZE), cgh);

    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;
      int lid = item.get_local_id(2);

      max_cache[lid] = ZERO;

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
      max_cache[lid] = new_u;

      // synchronize threads in block to ensure all velocities stored
      item.barrier(sycl::access::fence_space::local_space);

      // calculate maximum for block
      int i = BLOCK_SIZE >> 1;
      while (i != 0) {
        if (lid < i) {
          max_cache[lid] = sycl::fmax(max_cache[lid], max_cache[lid + i]);
        }
        item.barrier(sycl::access::fence_space::local_space);
        i >>= 1;
      }

      // store block's maximum
      if (lid == 0) {
        max_u[item.get_group_linear_id()] = max_cache[0];
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
} // end calculate_u

///////////////////////////////////////////////////////////////////////////////

void calculate_v (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const Real dt,
    const Real*__restrict__ G, 
    const Real*__restrict__ pres_red,
    const Real*__restrict__ pres_black, 
          Real*__restrict__ v,
          Real*__restrict__ max_v)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<Real, 1> max_cache (sycl::range<1>(BLOCK_SIZE), cgh);

    auto kfn = [=] (sycl::nd_item<3> item) {
      int row = item.get_global_id(2) + 1;
      int col = item.get_global_id(1) + 1;
      int lid = item.get_local_id(2);

      max_cache[lid] = ZERO;

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
      max_cache[lid] = new_v;

      // synchronize threads in block to ensure all velocities stored
      item.barrier(sycl::access::fence_space::local_space);

      // calculate maximum for block
      int i = BLOCK_SIZE >> 1;
      while (i != 0) {
        if (lid < i) {
          max_cache[lid] = sycl::fmax(max_cache[lid], max_cache[lid + i]);
        }
        item.barrier(sycl::access::fence_space::local_space);
        i >>= 1;
      }

      // store block's summed residuals
      if (lid == 0) {
        max_v[item.get_group_linear_id()] = max_cache[0];
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
} // end calculate_v

