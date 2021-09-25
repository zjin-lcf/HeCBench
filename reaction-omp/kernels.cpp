/**
 * @brief      Calculate Gray-Scott reaction rate
 */
void reaction_gray_scott(
    const float *__restrict fx, 
    const float *__restrict fy, 
    float *__restrict drx, 
    float *__restrict dry,
    const unsigned int ncells,
    const float d_c1,
    const float d_c2) 
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for(int i = 0; i < ncells; i ++) {
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
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils) 
{
  #pragma omp target teams num_teams(my * mz / pencils) thread_limit(mx * pencils)
  {
    float s_f[256];
    #pragma omp parallel
    {
      const int offset = 1;
      int threadIdx_x = omp_get_thread_num() % mx;
      int threadIdx_y = omp_get_thread_num() / mx;
      int blockIdx_x = omp_get_team_num() % (my / pencils);
      int blockIdx_y = omp_get_team_num() / (my / pencils);

      int i   = threadIdx_x;
      int j   = blockIdx_x * pencils + threadIdx_y;
      int k   = blockIdx_y;
      int si  = i + offset;  // local i for shared memory access + halo offset
      int sj  = threadIdx_y; // local j for shared memory access

      int globalIdx = k * mx * my + j * mx + i;

      s_f[sj * (mx + 2 * offset) + si] = f[globalIdx];

      #pragma omp barrier

      // fill in periodic images in shared memory array
      if (i < offset) {
        s_f[sj * (mx + 2 * offset) + si - offset]  = s_f[sj * (mx + 2 * offset) + si + mx - offset];
        s_f[sj * (mx + 2 * offset) + si + mx] = s_f[sj * (mx + 2 * offset) + si];
      }

      #pragma omp barrier

      df[globalIdx] = s_f[sj * (mx + 2 * offset) + si + 1] - 2.f * s_f[sj * (mx + 2 * offset) + si] + s_f[sj * (mx + 2 * offset) + si - 1];
    }
  }
}

/**
 * @brief      Calculate second derivative in x direction with zero-flux boundary conditions
 */
void derivative_x2_zeroflux(
    const float *__restrict f, 
    float *__restrict df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils) 
{
  #pragma omp target teams num_teams(my * mz / pencils) thread_limit(mx * pencils)
  {
    float s_f[256];
    #pragma omp parallel
    {
      int threadIdx_x = omp_get_thread_num() % mx;
      int threadIdx_y = omp_get_thread_num() / mx;
      int blockIdx_x = omp_get_team_num() % (my / pencils);
      int blockIdx_y = omp_get_team_num() / (my / pencils);

      int i   = threadIdx_x;
      int j   = blockIdx_x * pencils + threadIdx_y;
      int k   = blockIdx_y;
      int sj  = threadIdx_y; // local j for shared memory access

      int globalIdx = k * mx * my + j * mx + i;

      s_f[sj * mx + i] = f[globalIdx];

      #pragma omp barrier

      if(i == 0) {
        df[globalIdx] = s_f[sj * mx + i + 1] - s_f[sj * mx + i];
      } else if(i == (mx - 1)) {
        df[globalIdx] = s_f[sj * mx + i - 1] - s_f[sj * mx + i];
      } else {
        df[globalIdx] = s_f[sj * mx + i + 1] - 2.f * s_f[sj * mx + i] + s_f[sj * mx + i - 1];
      }
    }
  }
}

/**
 * @brief      Calculate second derivative in y direction with periodic boundary conditions
 */
void derivative_y2_pbc(
    const float *__restrict f, 
    float *__restrict df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils) 
{
  #pragma omp target teams num_teams(mx * mz / pencils) thread_limit(my * pencils)
  {
    float s_f[256];
    #pragma omp parallel
    {
      int threadIdx_x = omp_get_thread_num() % pencils;
      int threadIdx_y = omp_get_thread_num() / pencils;
      int blockIdx_x = omp_get_team_num() % (mx / pencils);
      int blockIdx_y = omp_get_team_num() / (mx / pencils);

      const int offset = 1;
      int i  = blockIdx_x * pencils + threadIdx_x;
      int j  = threadIdx_y;
      int k  = blockIdx_y;
      int si = threadIdx_x;
      int sj = j + offset;

      int globalIdx = k * mx * my + j * mx + i;

      s_f[sj * pencils + si] = f[globalIdx];

      #pragma omp barrier

      // fill in periodic images in shared memory array
      if (j < offset) {
        s_f[(sj - offset) * pencils + si]  = s_f[(sj + my - offset) * pencils + si];
        s_f[(sj + my) * pencils + si] = s_f[sj * pencils + si];
      }

      #pragma omp barrier

      df[globalIdx] = s_f[(sj+1) * pencils + si] - 2.f * s_f[sj * pencils + si] + s_f[(sj-1) * pencils + si];
    }
  }
}

/**
 * @brief      Calculate second derivative in y direction with zero-flux  boundary conditions
 */
void derivative_y2_zeroflux(
    const float *__restrict f, 
    float *__restrict df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils) 
{
  #pragma omp target teams num_teams(mx * mz / pencils) thread_limit(my * pencils)
  {
    float s_f[256];
    #pragma omp parallel
    {
      int threadIdx_x = omp_get_thread_num() % pencils;
      int threadIdx_y = omp_get_thread_num() / pencils;
      int blockIdx_x = omp_get_team_num() % (mx / pencils);
      int blockIdx_y = omp_get_team_num() / (mx / pencils);

      int i  = blockIdx_x * pencils + threadIdx_x;
      int j  = threadIdx_y;
      int k  = blockIdx_y;
      int si = threadIdx_x;

      int globalIdx = k * mx * my + j * mx + i;

      s_f[j * pencils + si] = f[globalIdx];

      #pragma omp barrier

      if(j == 0) {
        df[globalIdx] = s_f[(j+1) * pencils + si] - s_f[j * pencils + si];
      } else if(j == (my - 1)) {
        df[globalIdx] = s_f[(j-1) * pencils + si] - s_f[j * pencils + si];
      } else {
        df[globalIdx] = s_f[(j+1) * pencils + si] - 2.f * s_f[j * pencils + si] + s_f[(j-1) * pencils + si];
      }
    }
  }
}

/**
 * @brief      Calculate second derivative in z direction with periodic boundary conditions
 */
void derivative_z2_pbc(
    const float *__restrict f,
    float *__restrict df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils) 
{
  #pragma omp target teams num_teams(mx * my / pencils) thread_limit(mz * pencils)
  {
    float s_f[256];
    #pragma omp parallel
    {
      int threadIdx_x = omp_get_thread_num() % pencils;
      int threadIdx_y = omp_get_thread_num() / pencils;
      int blockIdx_x = omp_get_team_num() % (mx / pencils);
      int blockIdx_y = omp_get_team_num() / (mx / pencils);

      const int offset = 1;
      int i  = blockIdx_x * pencils + threadIdx_x;
      int j  = blockIdx_y;
      int k  = threadIdx_y;
      int si = threadIdx_x;
      int sk = k + offset; // halo offset

      int globalIdx = k * mx * my + j * mx + i;

      s_f[sk * pencils + si] = f[globalIdx];

      #pragma omp barrier

      // fill in periodic images in shared memory array
      if (k < offset) {
        s_f[(sk - offset) * pencils + si]  = s_f[(sk + mz - offset) * pencils + si];
        s_f[(sk + mz) * pencils + si] = s_f[sk * pencils + si];
      }

      #pragma omp barrier

      df[globalIdx] = s_f[(sk+1) * pencils + si] - 2.f * s_f[sk * pencils + si] + s_f[(sk-1) * pencils + si];
    }
  }
}

/**
 * @brief      Calculate second derivative in z direction with zero-flux boundary conditions
 */
void derivative_z2_zeroflux(
    const float *__restrict f, 
    float *__restrict df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils) 
{
  #pragma omp target teams num_teams(mx * my / pencils) thread_limit(mz * pencils)
  {
    float s_f[256];
    #pragma omp parallel
    {
      int threadIdx_x = omp_get_thread_num() % pencils;
      int threadIdx_y = omp_get_thread_num() / pencils;
      int blockIdx_x = omp_get_team_num() % (mx / pencils);
      int blockIdx_y = omp_get_team_num() / (mx / pencils);

      int i  = blockIdx_x * pencils + threadIdx_x;
      int j  = blockIdx_y;
      int k  = threadIdx_y;
      int si = threadIdx_x;

      int globalIdx = k * mx * my + j * mx + i;

      s_f[k * pencils + si] = f[globalIdx];

      #pragma omp barrier

      if(k == 0) {
        df[globalIdx] = s_f[(k+1) * pencils + si] - s_f[k * pencils + si];
      } else if(k == (mz - 1)) {
        df[globalIdx] = s_f[(k-1) * pencils + si] - s_f[k * pencils + si];
      } else {
        df[globalIdx] = s_f[(k+1) * pencils + si] - 2.f * s_f[k * pencils + si] + s_f[(k-1) * pencils + si];
      }
    }
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
    const unsigned int ncells, 
    const float d_diffcon) 
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for(int i = 0; i < ncells; i ++) {
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
    const unsigned int ncells,
    const float d_dt)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for(int i = 0; i < ncells; i ++) {
    x[i] += (ddx[i] + drx[i]) * d_dt;
    y[i] += (ddy[i] + dry[i]) * d_dt;
  }
}
