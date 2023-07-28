#include <hip/hip_runtime.h>
#include "XSbench_header.h"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor CPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, XSBench will only run the baseline implementation. Optimized variants
// are not yet implemented in this CUDA port.
////////////////////////////////////////////////////////////////////////////////////
__global__ void lookup (
    const int *__restrict__ num_nucs,
    const double *__restrict__ concs,
    const int *__restrict__ mats, 
    const NuclideGridPoint *__restrict__ nuclide_grid,
    int*__restrict__  verification,
    const double *__restrict__ unionized_energy_array,
    const int *__restrict__ index_grid,
    const int n_lookups,
    const long n_isotopes, 
    const long n_gridpoints,
    const int grid_type,
    const int hash_bins,
    const int max_num_nucs ) {

  // get the index to operate on, first dimemsion
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n_lookups) {

    // Set the initial seed value
    uint64_t seed = STARTING_SEED;  

    // Forward seed to lookup index (we need 2 samples per lookup)
    seed = fast_forward_LCG(seed, 2*i);

    // Randomly pick an energy and material for the particle
    double p_energy = LCG_random_double(&seed);
    int mat         = pick_mat(&seed); 

    // debugging
    //printf("E = %lf mat = %d\n", p_energy, mat);

    double macro_xs_vector[5] = {0};

    // Perform macroscopic Cross Section Lookup
    calculate_macro_xs(
        p_energy,     // Sampled neutron energy (in lethargy)
        mat,          // Sampled material type index neutron is in
        n_isotopes,   // Total number of isotopes in simulation
        n_gridpoints, // Number of gridpoints per isotope in simulation
        num_nucs,     // 1-D array with number of nuclides per material
        concs,        // Flattened 2-D array with concentration of each nuclide in each material
        unionized_energy_array, // 1-D Unionized energy array
        index_grid,   // Flattened 2-D grid holding indices into nuclide grid for each unionized energy level
        nuclide_grid, // Flattened 2-D grid holding energy levels and XS_data for all nuclides in simulation
        mats,         // Flattened 2-D array with nuclide indices defining composition of each type of material
        macro_xs_vector, // 1-D array with result of the macroscopic cross section (5 different reaction channels)
        grid_type,    // Lookup type (nuclide, hash, or unionized)
        hash_bins,    // Number of hash bins used (if using hash lookup type)
        max_num_nucs  // Maximum number of nuclides present in any material
     );

    // For verification, and to prevent the compiler from optimizing
    // all work out, we interrogate the returned macro_xs_vector array
    // to find its maximum value index, then increment the verification
    // value by that index. In this implementation, we store to a global
    // array that will get tranferred back and reduced on the host.
    double max = -1.0;
    int max_idx = 0;
    for(int j = 0; j < 5; j++ )
    {
      if( macro_xs_vector[j] > max )
      {
        max = macro_xs_vector[j];
        max_idx = j;
      }
    }
    verification[i] = max_idx+1;
  }
}


unsigned long long
run_event_based_simulation(Inputs in, SimulationData SD,
                           int mype, double *kernel_time)
{

  ////////////////////////////////////////////////////////////////////////////////
  // SUMMARY: Simulation Data Structure Manifest for "SD" Object
  // Here we list all heap arrays (and lengths) in SD that would need to be
  // offloaded manually if using an accelerator with a seperate memory space
  ////////////////////////////////////////////////////////////////////////////////
  // int * num_nucs;                     // Length = length_num_nucs;
  // double * concs;                     // Length = length_concs
  // int * mats;                         // Length = length_mats
  // double * unionized_energy_array;    // Length = length_unionized_energy_array
  // int * index_grid;                   // Length = length_index_grid
  // NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
  // 
  // Note: "unionized_energy_array" and "index_grid" can be of zero length
  //        depending on lookup method.
  //
  // Note: "Lengths" are given as the number of objects in the array, not the
  //       number of bytes.
  ////////////////////////////////////////////////////////////////////////////////

  if(mype==0) printf("Beginning event based simulation...\n");

  // Let's create an extra verification array to reduce manually later on
  if( mype == 0 )
     printf("Allocating an additional %.1lf MB of memory for verification arrays...\n",
            in.lookups * sizeof(int) /1024.0/1024.0);

  int * verification_h = (int *) malloc(in.lookups * sizeof(int));

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  if(mype == 0 ) printf("Running on: %s\n", devProp.name);
  if(mype == 0 ) printf("Initializing device buffers and JIT compiling kernel...\n");

  ////////////////////////////////////////////////////////////////////////////////
  // Create Device Buffers
  ////////////////////////////////////////////////////////////////////////////////

  int *verification_d = nullptr;
  int *mats_d = nullptr ;
  int *num_nucs_d = nullptr;
  double *concs_d = nullptr;
  NuclideGridPoint *nuclide_grid_d = nullptr;

  //buffer<int, 1> num_nucs_d(SD.num_nucs,SD.length_num_nucs);
  hipMalloc((void**)&num_nucs_d, sizeof(int) * SD.length_num_nucs);
  hipMemcpy(num_nucs_d, SD.num_nucs, sizeof(int) * SD.length_num_nucs, hipMemcpyHostToDevice);

  //buffer<double, 1> concs_d(SD.concs, SD.length_concs);
  hipMalloc((void**)&concs_d, sizeof(double) * SD.length_concs);
  hipMemcpy(concs_d, SD.concs, sizeof(double) * SD.length_concs, hipMemcpyHostToDevice);

  //buffer<int, 1> mats_d(SD.mats, SD.length_mats);
  hipMalloc((void**)&mats_d, sizeof(int) * SD.length_mats);
  hipMemcpy(mats_d, SD.mats, sizeof(int) * SD.length_mats, hipMemcpyHostToDevice);

  //buffer<NuclideGridPoint, 1> nuclide_grid_d(SD.nuclide_grid, SD.length_nuclide_grid);
  hipMalloc((void**)&nuclide_grid_d, sizeof(NuclideGridPoint) * SD.length_nuclide_grid);
  hipMemcpy(nuclide_grid_d, SD.nuclide_grid, sizeof(NuclideGridPoint) * SD.length_nuclide_grid, hipMemcpyHostToDevice);

  //buffer<int, 1> verification_d(verification_h, in.lookups);
  hipMalloc((void**)&verification_d, sizeof(int) * in.lookups);

  // These two are a bit of a hack. Sometimes they are empty buffers (if using hash or nuclide
  // grid methods). OpenCL will throw an example when we try to create an empty buffer. So, we
  // will just allocate some memory for them and move them as normal. The rest of our code
  // won't actually use them if they aren't needed, so this is safe. Probably a cleaner way
  // of doing this.
  if( SD.length_unionized_energy_array == 0 )
  {
    SD.length_unionized_energy_array = 1;
    SD.unionized_energy_array = (double *) malloc(sizeof(double));
  }
  //buffer<double,1> unionized_energy_array_d(SD.unionized_energy_array, SD.length_unionized_energy_array);
  double *unionized_energy_array_d = nullptr;
  hipMalloc((void**)&unionized_energy_array_d, sizeof(double) * SD.length_unionized_energy_array);
  hipMemcpy(unionized_energy_array_d, SD.unionized_energy_array, 
      sizeof(double) * SD.length_unionized_energy_array, hipMemcpyHostToDevice);

  if( SD.length_index_grid == 0 )
  {
    SD.length_index_grid = 1;
    SD.index_grid = (int *) malloc(sizeof(int));
  }

  //buffer<int, 1> index_grid_d(SD.index_grid, (unsigned long long ) SD.length_index_grid);
  int *index_grid_d = nullptr;
  hipMalloc((void**)&index_grid_d, sizeof(int) * (unsigned long long)SD.length_index_grid);
  hipMemcpy(index_grid_d, SD.index_grid, sizeof(int) * (unsigned long long )SD.length_index_grid, hipMemcpyHostToDevice);

  hipDeviceSynchronize();

  ////////////////////////////////////////////////////////////////////////////////
  // Define Device Kernel
  ////////////////////////////////////////////////////////////////////////////////
  dim3 grids  ((in.lookups + 255) / 256);
  dim3 blocks (256);

  double kstart = get_time();

  for (int i = 0; i < in.kernel_repeat; i++) { 
    lookup<<< grids, blocks >>> (
        num_nucs_d, concs_d, mats_d, 
        nuclide_grid_d, verification_d, unionized_energy_array_d,
        index_grid_d, in.lookups, in.n_isotopes, in.n_gridpoints, 
        in.grid_type, in.hash_bins, SD.max_num_nucs );
  }

  hipDeviceSynchronize();
  double kstop = get_time();
  *kernel_time = (kstop - kstart) / in.kernel_repeat;

  hipMemcpy(verification_h, verification_d, sizeof(int) * in.lookups, hipMemcpyDeviceToHost);

  hipFree(verification_d);
  hipFree(mats_d);
  hipFree(num_nucs_d);
  hipFree(concs_d);
  hipFree(nuclide_grid_d);
  hipFree(unionized_energy_array_d);
  hipFree(index_grid_d);

  // Host reduces the verification array
  unsigned long long verification_scalar = 0;
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += verification_h[i];

  if( SD.length_unionized_energy_array == 0 ) free(SD.unionized_energy_array);
  if( SD.length_index_grid == 0 ) free(SD.index_grid);
  free(verification_h);

  return verification_scalar;
}


// binary search for energy on unionized energy grid
// returns lower index
template <class T>
__device__
long grid_search( long n, double quarry, T A)
{
  long lowerLimit = 0;
  long upperLimit = n-1;
  long examinationPoint;
  long length = upperLimit - lowerLimit;

  while( length > 1 )
  {
    examinationPoint = lowerLimit + ( length / 2 );

    if( A[examinationPoint] > quarry )
      upperLimit = examinationPoint;
    else
      lowerLimit = examinationPoint;

    length = upperLimit - lowerLimit;
  }

  return lowerLimit;
}

// Calculates the microscopic cross section for a given nuclide & energy
template <class Double_Type, class Int_Type, class NGP_Type>
__device__
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
    long n_gridpoints,
    Double_Type  egrid, Int_Type  index_data,
    NGP_Type  nuclide_grids,
    long idx, double *  xs_vector, int grid_type, int hash_bins ){
  // Variables
  double f;
  NuclideGridPoint low, high;
  long low_idx, high_idx;

  // If using only the nuclide grid, we must perform a binary search
  // to find the energy location in this particular nuclide's grid.
  if( grid_type == NUCLIDE )
  {
    // Perform binary search on the Nuclide Grid to find the index
    long offset = nuc * n_gridpoints;
    idx = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset, offset + n_gridpoints-1);

    // pull ptr from nuclide grid and check to ensure that
    // we're not reading off the end of the nuclide's grid
    if( idx == n_gridpoints - 1 )
      low_idx = idx - 1;
    else
      low_idx = idx;
  }
  else if( grid_type == UNIONIZED) // Unionized Energy Grid - we already know the index, no binary search needed.
  {
    // pull ptr from energy grid and check to ensure that
    // we're not reading off the end of the nuclide's grid
    if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
      low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1;
    else
    {
      low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc];
    }
  }
  else // Hash grid
  {
    // load lower bounding index
    int u_low = index_data[idx * n_isotopes + nuc];

    // Determine higher bounding index
    int u_high;
    if( idx == hash_bins - 1 )
      u_high = n_gridpoints - 1;
    else
      u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;

    // Check edge cases to make sure energy is actually between these
    // Then, if things look good, search for gridpoint in the nuclide grid
    // within the lower and higher limits we've calculated.
    double e_low  = nuclide_grids[nuc*n_gridpoints + u_low].energy;
    double e_high = nuclide_grids[nuc*n_gridpoints + u_high].energy;
    long lower;
    if( p_energy <= e_low )
      lower = nuc*n_gridpoints;
    else if( p_energy >= e_high )
      lower = nuc*n_gridpoints + n_gridpoints - 1;
    else
    {
      long offset = nuc*n_gridpoints;
      lower = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset+u_low, offset+u_high);
    }

    if( (lower % n_gridpoints) == n_gridpoints - 1 )
      low_idx = lower - 1;
    else
      low_idx = lower;
  }

  high_idx = low_idx + 1;
  low = nuclide_grids[low_idx];
  high = nuclide_grids[high_idx];

  // calculate the re-useable interpolation factor
  f = (high.energy - p_energy) / (high.energy - low.energy);

  // Total XS
  xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);

  // Elastic XS
  xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);

  // Absorbtion XS
  xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);

  // Fission XS
  xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);

  // Nu Fission XS
  xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
}

// Calculates macroscopic cross section based on a given material & energy 
template <class Double_Type, class Int_Type, class NGP_Type, class E_GRID_TYPE, class INDEX_TYPE>
__device__
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
    long n_gridpoints, Int_Type  num_nucs,
    Double_Type  concs,
    E_GRID_TYPE  egrid, INDEX_TYPE  index_data,
    NGP_Type  nuclide_grids,
    Int_Type  mats,
    double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
  int p_nuc; // the nuclide we are looking up
  long idx = -1;  
  double conc; // the concentration of the nuclide in the material

  // cleans out macro_xs_vector
  for( int k = 0; k < 5; k++ )
    macro_xs_vector[k] = 0;

  // If we are using the unionized energy grid (UEG), we only
  // need to perform 1 binary search per macroscopic lookup.
  // If we are using the nuclide grid search, it will have to be
  // done inside of the "calculate_micro_xs" function for each different
  // nuclide in the material.
  if( grid_type == UNIONIZED )
    idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);  
  else if( grid_type == HASH )
  {
    double du = 1.0 / hash_bins;
    idx = p_energy / du;
  }

  // Once we find the pointer array on the UEG, we can pull the data
  // from the respective nuclide grids, as well as the nuclide
  // concentration data for the material
  // Each nuclide from the material needs to have its micro-XS array
  // looked up & interpolatied (via calculate_micro_xs). Then, the
  // micro XS is multiplied by the concentration of that nuclide
  // in the material, and added to the total macro XS array.
  // (Independent -- though if parallelizing, must use atomic operations
  //  or otherwise control access to the xs_vector and macro_xs_vector to
  //  avoid simulataneous writing to the same data structure)
  for( int j = 0; j < num_nucs[mat]; j++ )
  {
    double xs_vector[5];
    p_nuc = mats[mat*max_num_nucs + j];
    conc = concs[mat*max_num_nucs + j];
    calculate_micro_xs( p_energy, p_nuc, n_isotopes,
        n_gridpoints, egrid, index_data,
        nuclide_grids, idx, xs_vector, grid_type, hash_bins );
    for( int k = 0; k < 5; k++ )
      macro_xs_vector[k] += xs_vector[k] * conc;
  }
}

// picks a material based on a probabilistic distribution
__device__
int pick_mat( unsigned long * seed )
{
  // I have a nice spreadsheet supporting these numbers. They are
  // the fractions (by volume) of material in the core. Not a 
  // *perfect* approximation of where XS lookups are going to occur,
  // but this will do a good job of biasing the system nonetheless.

  // Also could be argued that doing fractions by weight would be 
  // a better approximation, but volume does a good enough job for now.

  double dist[12];
  dist[0]  = 0.140;  // fuel
  dist[1]  = 0.052;  // cladding
  dist[2]  = 0.275;  // cold, borated water
  dist[3]  = 0.134;  // hot, borated water
  dist[4]  = 0.154;  // RPV
  dist[5]  = 0.064;  // Lower, radial reflector
  dist[6]  = 0.066;  // Upper reflector / top plate
  dist[7]  = 0.055;  // bottom plate
  dist[8]  = 0.008;  // bottom nozzle
  dist[9]  = 0.015;  // top nozzle
  dist[10] = 0.025;  // top of fuel assemblies
  dist[11] = 0.013;  // bottom of fuel assemblies

  double roll = LCG_random_double(seed);

  // makes a pick based on the distro
  for( int i = 0; i < 12; i++ )
  {
    double running = 0;
    for( int j = i; j > 0; j-- )
      running += dist[j];
    if( roll < running )
      return i;
  }

  return 0;
}

__host__ __device__
double LCG_random_double(uint64_t * seed)
{
  // LCG parameters
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}  

__host__ __device__
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
  // LCG parameters
  const uint64_t m = 9223372036854775808ULL; // 2^63
  uint64_t a = 2806196910506780709ULL;
  uint64_t c = 1ULL;

  n = n % m;

  uint64_t a_new = 1;
  uint64_t c_new = 0;

  while(n > 0) 
  {
    if(n & 1)
    {
      a_new *= a;
      c_new = c_new * a + c;
    }
    c *= (a + 1);
    a *= a;

    n >>= 1;
  }

  return (a_new * seed + c_new) % m;
}
