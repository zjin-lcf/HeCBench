/**
 * @brief      Calculate Gray-Scott reaction rate
 */
void reaction_gray_scott(
    const float *__restrict fx,
    const float *__restrict fy,
    float *__restrict drx,
    float *__restrict dry,
    sycl::nd_item<1> &item,
    const unsigned int ncells,
    const float d_c1,
    const float d_c2)
{
  int index = item.get_global_id(0);
  int stride = item.get_local_range(0) * item.get_group_range(0);

  for(int i = index; i < ncells; i += stride) {
    float r = fx[i] * fy[i] * fy[i];
    drx[i] = -r + d_c1 * (1.f - fx[i]);
    dry[i] = r - (d_c1 + d_c2) * fy[i];
  }
}

/**
 * @brief      Calculate second derivative in x direction with periodic boundary conditions
 */
void derivative_x2_pbc(
    const float *__restrict f,
    float *__restrict df,
    float *__restrict s_f, // 2-wide halo
    sycl::nd_item<2> &item,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int pencils)
{
  const int offset = 1;
  int i   = item.get_local_id(1);
  int si  = i + offset;           // local i for shared memory access + halo offset
  int sj  = item.get_local_id(0); // local j for shared memory access
  int j   = item.get_group(1) * item.get_local_range(0) + sj;
  int k   = item.get_group(0);

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj * (mx + 2 * offset) + si] = f[globalIdx];

  item.barrier(sycl::access::fence_space::local_space);

  // fill in periodic images in shared memory array
  if (i < offset) {
    s_f[sj * (mx + 2 * offset) + si - offset]  = s_f[sj * (mx + 2 * offset) + si + mx - offset];
    s_f[sj * (mx + 2 * offset) + si + mx] = s_f[sj * (mx + 2 * offset) + si];
  }

  item.barrier(sycl::access::fence_space::local_space);

  df[globalIdx] = s_f[sj * (mx + 2 * offset) + si + 1] - 2.f * s_f[sj * (mx + 2 * offset) + si] + s_f[sj * (mx + 2 * offset) + si - 1];
}

/**
 * @brief      Calculate second derivative in x direction with zero-flux boundary conditions
 */
void derivative_x2_zeroflux(
    const float *__restrict f,
    float *__restrict df,
    float *__restrict s_f,
    sycl::nd_item<2> &item,
    const unsigned int mx,
    const unsigned int my)
{
  int i   = item.get_local_id(1);
  int sj  = item.get_local_id(0); // local j for shared memory access
  int j   = item.get_group(1) * item.get_local_range(0) + sj;
  int k   = item.get_group(0);

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj * mx + i] = f[globalIdx];

  item.barrier(sycl::access::fence_space::local_space);

  if(i == 0) {
    df[globalIdx] = s_f[sj * mx + i + 1] - s_f[sj * mx + i];
  } else if(i == (mx - 1)) {
    df[globalIdx] = s_f[sj * mx + i - 1] - s_f[sj * mx + i];
  } else {
    df[globalIdx] = s_f[sj * mx + i + 1] - 2.f * s_f[sj * mx + i] + s_f[sj * mx + i - 1];
  }
}

/**
 * @brief      Calculate second derivative in y direction with periodic boundary conditions
 */
void derivative_y2_pbc(
    const float *__restrict f,
    float *__restrict df,
    float *__restrict s_f,
    sycl::nd_item<2> &item,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int pencils)
{
  const int offset = 1;

  int i  = item.get_global_id(1);
  int j  = item.get_local_id(0);
  int k  = item.get_group(0);
  int si = item.get_local_id(1);
  int sj = j + offset;
  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj * pencils + si] = f[globalIdx];

  item.barrier(sycl::access::fence_space::local_space);

  // fill in periodic images in shared memory array
  if (j < offset) {
    s_f[(sj - offset) * pencils + si]  = s_f[(sj + my - offset) * pencils + si];
    s_f[(sj + my) * pencils + si] = s_f[sj * pencils + si];
  }

  item.barrier(sycl::access::fence_space::local_space);

  df[globalIdx] = s_f[(sj+1) * pencils + si] - 2.f * s_f[sj * pencils + si] + s_f[(sj-1) * pencils + si];
}

/**
 * @brief      Calculate second derivative in y direction with zero-flux  boundary conditions
 */
void derivative_y2_zeroflux(
    const float *__restrict f,
    float *__restrict df,
    float *__restrict s_f,
    sycl::nd_item<2> &item,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int pencils)
{
  int i  = item.get_global_id(1);
  int j  = item.get_local_id(0);
  int k  = item.get_group(0);
  int si = item.get_local_id(1);
  int globalIdx = k * mx * my + j * mx + i;

  s_f[j * pencils + si] = f[globalIdx];

  item.barrier(sycl::access::fence_space::local_space);

  if(j == 0) {
    df[globalIdx] = s_f[(j+1) * pencils + si] - s_f[j * pencils + si];
  } else if(j == (my - 1)) {
    df[globalIdx] = s_f[(j-1) * pencils + si] - s_f[j * pencils + si];
  } else {
    df[globalIdx] = s_f[(j+1) * pencils + si] - 2.f * s_f[j * pencils + si] + s_f[(j-1) * pencils + si];
  }
}

/**
 * @brief      Calculate second derivative in z direction with periodic boundary conditions
 */
void derivative_z2_pbc(
    const float *__restrict f,
    float *__restrict df,
    float *__restrict s_f,
    sycl::nd_item<2> &item,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils)
{
  const int offset = 1;
  int i  = item.get_global_id(1);
  int j  = item.get_group(0);
  int k  = item.get_local_id(0);
  int si = item.get_local_id(1);
  int sk = k + offset; // halo offset
  int globalIdx = k * mx * my + j * mx + i;

  s_f[sk * pencils + si] = f[globalIdx];

  item.barrier(sycl::access::fence_space::local_space);

  // fill in periodic images in shared memory array
  if (k < offset) {
    s_f[(sk - offset) * pencils + si]  = s_f[(sk + mz - offset) * pencils + si];
    s_f[(sk + mz) * pencils + si] = s_f[sk * pencils + si];
  }

  item.barrier(sycl::access::fence_space::local_space);

  df[globalIdx] = s_f[(sk+1) * pencils + si] - 2.f * s_f[sk * pencils + si] + s_f[(sk-1) * pencils + si];
}

/**
 * @brief      Calculate second derivative in z direction with zero-flux boundary conditions
 */
void derivative_z2_zeroflux(
    const float *__restrict f,
    float *__restrict df,
    float *__restrict s_f,
    sycl::nd_item<2> &item,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils)
{
  int i  = item.get_global_id(1);
  int j  = item.get_group(0);
  int k  = item.get_local_id(0);
  int si = item.get_local_id(1);
  int globalIdx = k * mx * my + j * mx + i;

  s_f[k * pencils + si] = f[globalIdx];

  item.barrier(sycl::access::fence_space::local_space);

  if(k == 0) {
    df[globalIdx] = s_f[(k+1) * pencils + si] - s_f[k * pencils + si];
  } else if(k == (mz - 1)) {
    df[globalIdx] = s_f[(k-1) * pencils + si] - s_f[k * pencils + si];
  } else {
    df[globalIdx] = s_f[(k+1) * pencils + si] - 2.f * s_f[k * pencils + si] + s_f[(k-1) * pencils + si];
  }
}

/**
 * @brief      Construct the Laplacian for a component
 */
void construct_laplacian(
    float *__restrict df,
    const float *__restrict dfx,
    const float *__restrict dfy,
    const float *__restrict dfz,
    sycl::nd_item<1> &item,
    const unsigned int ncells,
    const float d_diffcon)
{
  int index = item.get_global_id(0);
  int stride = item.get_local_range(0) * item.get_group_range(0);

  for(int i = index; i < ncells; i += stride) {
    df[i] = d_diffcon * (dfx[i] + dfy[i] + dfz[i]);
  }
}

/**
 * @brief      Perform time-step integration
 *
 * @param      x     pointer to concentration of A
 * @param      y     pointer to concentration of B
 * @param[in]  ddx   diffusion of component A
 * @param[in]  ddy   diffusion of component B
 * @param[in]  drx   reaction of component A
 * @param[in]  dry   reaction of component B
 */
void update(
    float *__restrict x,
    float *__restrict y,
    const float *__restrict ddx,
    const float *__restrict ddy,
    const float *__restrict drx,
    const float *__restrict dry,
    sycl::nd_item<1> &item,
    const unsigned int ncells,
    const float d_dt)
{
  int index = item.get_global_id(0);
  int stride = item.get_local_range(0) * item.get_group_range(0);

  for(int i = index; i < ncells; i += stride) {
    x[i] += (ddx[i] + drx[i]) * d_dt;
    y[i] += (ddy[i] + dry[i]) * d_dt;
  }
}
