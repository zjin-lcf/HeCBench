/***************************************************************************
 *
Copyright 2020 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 * motionsim.cpp
 *
 * Description:
 *   This code sample will implement a simple example of a Monte Carlo
 *   simulation of the diffusion of water molecules in tissue.
 *
 **************************************************************************/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iomanip>

// Helper functions

// This function displays correct usage and parameters
void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " <Numbeof Iterations> " << std::endl << std::endl;
}

// This function prints a 2D matrix
template <typename T>
void print_matrix(T** matrix, size_t size_X, size_t size_Y) {
  std::cout << std::endl;
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      std::cout << std::setw(3) << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

// This function prints a vector
template <typename T>
void print_vector(T* vector, size_t n) {
  std::cout << std::endl;
  for (size_t i = 0; i < n; ++i) {
    std::cout << vector[i] << " ";
  }
  std::cout << std::endl;
}

void Simulation(float* a_particleX, float* a_particleY,
		const float* a_randomX, const float* a_randomY, 
		size_t *a_map, const size_t n_particles, unsigned int nIterations, int grid_size, float radius ,
		sycl::nd_item<3> item_ct1) {
  size_t ii = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    if (ii >= n_particles) return;
    size_t randnumX = 0;
    size_t randnumY = 0;
    float displacementX = 0.0f;
    float displacementY = 0.0f;

    // Start iterations
    // Each iteration:
    //  1. Updates the position of all water molecules
    //  2. Checks if water molecule is inside a cell or not.
    //  3. Updates counter in cells array
    size_t iter = 0;
    while (iter < nIterations) {
	    // Computes random displacement for each molecule
	    // This example shows random distances between
	    // -0.05 units and 0.05 units in both X and Y directions
	    // Moves each water molecule by a random vector

	    randnumX = a_randomX[iter * n_particles + ii];
	    randnumY = a_randomY[iter * n_particles + ii];

	    // Transform the scaled random numbers into small displacements
	    displacementX = (float)randnumX / 1000.0f - 0.0495;
	    displacementY = (float)randnumY / 1000.0f - 0.0495;

	    // Move particles using random displacements
	    a_particleX[ii] += displacementX;
	    a_particleY[ii] += displacementY;

	    // Compute distances from particle position to grid point
    float dX = a_particleX[ii] - sycl::trunc(a_particleX[ii]);
    float dY = a_particleY[ii] - sycl::trunc(a_particleY[ii]);

            // Compute grid point indices
    int iX = sycl::floor(a_particleX[ii]);
    int iY = sycl::floor(a_particleY[ii]);

            // Check if particle is still in computation grid
	    if ((a_particleX[ii] < grid_size) &&
		(a_particleY[ii] < grid_size) && (a_particleX[ii] >= 0) &&
		(a_particleY[ii] >= 0)) {
		    // Check if particle is (or remained) inside cell.
		    // Increment cell counter in map array if so
		    if ((dX * dX + dY * dY <= radius * radius)) {
			    // The map array is organized as (particle, y, x)
			    a_map[ii * grid_size * grid_size + iY * grid_size + iX]++;
		    }
	    }

	    iter++;

    }  // Next iteration
}

// This function distributes simulation work across workers
void motion_device(float *particleX, float *particleY, float *randomX,
                   float *randomY, int **grid, int grid_size,
                   size_t n_particles, unsigned int nIterations, float radius,
                   size_t *map) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  dpct::device_info devProp;
  dpct::dev_mgr::instance().get_device(0).get_device_info(devProp);

  std::cout << " Running on:: " << devProp.get_name() << std::endl;
  std::cout << " The Device Max Work Group Size is : "
            << devProp.get_max_work_group_size() << std::endl;
  std::cout << " The number of iterations is : " << nIterations << std::endl;
  std::cout << " The number of particles is : " << n_particles << std::endl;

  // Set the seed for rand() function.
  // Use a fixed seed for reproducibility/debugging
  srand(17);
  
  // Scale of random numbers
  const size_t scale = 100;

  // Compute vectors of random values for X and Y directions
  for (size_t i = 0; i < n_particles * nIterations; i++) {
    randomX[i] = rand() % scale;
    randomY[i] = rand() % scale;
  }

  float *d_randomX;
  float *d_randomY;
  float *d_particleX;
  float *d_particleY;
  size_t *d_map;
  d_randomX = (float *)sycl::malloc_device(
      sizeof(float) * n_particles * nIterations, q_ct1);
  d_randomY = (float *)sycl::malloc_device(
      sizeof(float) * n_particles * nIterations, q_ct1);
  d_particleX = sycl::malloc_device<float>(n_particles, q_ct1);
  d_particleY = sycl::malloc_device<float>(n_particles, q_ct1);
  size_t MAP_SIZE = n_particles * grid_size * grid_size;
  d_map = sycl::malloc_device<size_t>(MAP_SIZE, q_ct1);

  q_ct1.memcpy(d_randomX, randomX, sizeof(float) * n_particles * nIterations)
      .wait();
  q_ct1.memcpy(d_randomY, randomY, sizeof(float) * n_particles * nIterations)
      .wait();
  q_ct1.memcpy(d_particleX, particleX, sizeof(float) * n_particles).wait();
  q_ct1.memcpy(d_particleY, particleY, sizeof(float) * n_particles).wait();
  q_ct1.memcpy(d_map, map, sizeof(size_t) * MAP_SIZE).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (n_particles + 255) / 256) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
          Simulation(d_particleX, d_particleY, d_randomX, d_randomY, d_map,
                     n_particles, nIterations, grid_size, radius, item_ct1);
        });
  });

  q_ct1.memcpy(map, d_map, sizeof(size_t) * MAP_SIZE).wait();
  // For every cell in the grid, add all the counters from different
  // particles (workers) which are stored in the 3rd dimension of the 'map'
  // array
  for (size_t i = 0; i < n_particles; ++i) {
    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        if (map[i * grid_size * grid_size + y * grid_size + x] > 0) {
          grid[y][x] += map[i * grid_size * grid_size + y * grid_size + x];
        }
      }
    }
  }  // End loop for number of particles

  sycl::free(d_randomX, q_ct1);
  sycl::free(d_randomY, q_ct1);
  sycl::free(d_particleX, q_ct1);
  sycl::free(d_particleY, q_ct1);
  sycl::free(d_map, q_ct1);
}  // End of function motion_device()

int main(int argc, char* argv[]) {
  // Cell and Particle parameters
  const size_t grid_size = 21;    // Size of square grid
  const size_t n_particles = 147456;  // Number of particles
  const float radius = 0.5;       // Cell radius = 0.5*(grid spacing)

  // Default number of operations
  size_t nIterations = 50;

  // Read command-line arguments
  try {
    nIterations = std::stoi(argv[1]);
  }

  catch (...) {
    usage(argv[0]);
    return 1;
  }

  // Allocate arrays

  // Stores a grid of cells
  int** grid = new int*[grid_size];
  for (size_t i = 0; i < grid_size; i++) grid[i] = new int[grid_size];

  // Stores all random numbers to be used in the simulation
  float* randomX = new float[n_particles * nIterations];
  float* randomY = new float[n_particles * nIterations];

  // Stores X and Y position of particles in the cell grid
  float* particleX = new float[n_particles];
  float* particleY = new float[n_particles];

  // 'map' array replicates grid to be used by each particle
  size_t MAP_SIZE = n_particles * grid_size * grid_size;
  size_t* map = new size_t[MAP_SIZE];

  // Initialize arrays
  for (size_t i = 0; i < n_particles; i++) {
    // Initial position of particles in cell grid
    particleX[i] = 10.0;
    particleY[i] = 10.0;

    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        map[i * grid_size * grid_size + y * grid_size + x] = 0;
      }
    }
  }

  for (size_t y = 0; y < grid_size; y++) {
    for (size_t x = 0; x < grid_size; x++) {
      grid[y][x] = 0;
    }
  }

  // Start timers
  auto start = std::chrono::steady_clock::now();

  // Call simulation function
  motion_device(particleX, particleY, randomX, randomY, grid, grid_size,
                n_particles, nIterations, radius, map);


  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  std::cout << std::endl;
  std::cout << "Time: " << time << std::endl;
  std::cout << std::endl;

  // Displays final grid only if grid small.
  if (grid_size <= 64) {
    std::cout << "\n ********************** OUTPUT GRID: " << std::endl;
    print_matrix<int>(grid, grid_size, grid_size);
  }

  // Cleanup
  for (size_t i = 0; i < grid_size; i++) delete grid[i];

  delete[] grid;
  delete[] particleX;
  delete[] particleY;
  delete[] randomX;
  delete[] randomY;
  delete[] map;

  return 0;
}
