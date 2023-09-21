void kernels(
  const int kernel_id, 
  const int num, 
  const int dim, 
  const int size_sigma, 
  const int size_hamiltonian, 
  const real_t hdt, 
  const real_2_t *__restrict sigma_in, 
        real_2_t *__restrict sigma_out,
  const real_2_t *__restrict hamiltonian)
{

#pragma omp target data map(alloc: sigma_out[0:size_sigma]) \
                        map(to: sigma_in[0:size_sigma], \
                                hamiltonian[0:size_hamiltonian])
{
  // benchmark loop
  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
    // clear output 
    #pragma omp target update to (sigma_out[0:size_sigma])

    // empty kernel
    switch(kernel_id) {
      case 0:  {
        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {}
        break;
      }

      // initial kernel
      case 1: {
        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          int sigma_id = gid * dim * dim;
          // compute commutator: -i * dt/hbar * (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
          for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
              real_2_t tmp;
              tmp.x = 0.0;
              tmp.y = 0.0;
              for (int k = 0; k < dim; ++k) {
                // z=(x,y), w=(u,v)  z*w = (xu-yv, xv+yu)
                tmp.x += (hamiltonian[i * dim + k].x * sigma_in[sigma_id + k * dim + j].x - 
                          sigma_in[sigma_id + i * dim + k].x * hamiltonian[k * dim + j].x);
                tmp.x -= (hamiltonian[i * dim + k].y * sigma_in[sigma_id + k * dim + j].y - 
                          sigma_in[sigma_id + i * dim + k].y * hamiltonian[k * dim + j].y);
                tmp.y += (hamiltonian[i * dim + k].x * sigma_in[sigma_id + k * dim + j].y - 
                          sigma_in[sigma_id + i * dim + k].x * hamiltonian[k * dim + j].y);
                tmp.y += (hamiltonian[i * dim + k].y * sigma_in[sigma_id + k * dim + j].x -
                          sigma_in[sigma_id + i * dim + k].y * hamiltonian[k * dim + j].x);
              }
              // multiply with -i * dt / hbar
              sigma_out[sigma_id + i * dim + j].x += hdt * tmp.y;
              sigma_out[sigma_id + i * dim + j].y -= hdt * tmp.x;
            }
          }
        }
        break;
      }

