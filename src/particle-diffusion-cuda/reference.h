void reference(float*__restrict__ a_particleX,
               float*__restrict__ a_particleY,
	       const float*__restrict__ a_randomX,
               const float*__restrict__ a_randomY, 
	       size_t *__restrict__ a_map,
               const size_t n_particles,
               unsigned int nIterations,
               int grid_size,
               float radius)
{
  for (size_t ii = 0; ii < n_particles; ii++) {
    // Start iterations
    // Each iteration:
    //  1. Updates the position of all water molecules
    //  2. Checks if water molecule is inside a cell or not.
    //  3. Updates counter in cells array
    size_t iter = 0;
    float pX = a_particleX[ii];
    float pY = a_particleY[ii];
    size_t map_base = ii * grid_size * grid_size;
    while (iter < nIterations) {
      // Computes random displacement for each molecule
      // This example shows random distances between
      // -0.05 units and 0.05 units in both X and Y directions
      // Moves each water molecule by a random vector

      float randnumX = a_randomX[iter * n_particles + ii];
      float randnumY = a_randomY[iter * n_particles + ii];

      // Transform the scaled random numbers into small displacements
      float displacementX = randnumX / 1000.0f - 0.0495f;
      float displacementY = randnumY / 1000.0f - 0.0495f;

      // Move particles using random displacements
      pX += displacementX;
      pY += displacementY;

      // Compute distances from particle position to grid point
      float dX = pX - truncf(pX);
      float dY = pY - truncf(pY);

      // Compute grid point indices
      int iX = floorf(pX);
      int iY = floorf(pY);

      // Check if particle is still in computation grid
      if ((pX < grid_size) && (pY < grid_size) && (pX >= 0) && (pY >= 0)) {
        // Check if particle is (or remained) inside cell.
        // Increment cell counter in map array if so
        if ((dX * dX + dY * dY <= radius * radius))
          // The map array is organized as (particle, y, x)
          a_map[map_base + iY * grid_size + iX]++;
      }

      iter++;

    }  // Next iteration

    a_particleX[ii] = pX;
    a_particleY[ii] = pY;
  }
}

void motion_host(float* particleX, float* particleY,
                   float* randomX, float* randomY, int** grid, size_t grid_size,
                   size_t n_particles, int nIterations, float radius,
                   size_t* map, int nRepeat) {
  reference(particleX, 
            particleY, 
            randomX, 
            randomY, 
            map, 
            n_particles,
            nIterations,
            grid_size,
            radius);
}
