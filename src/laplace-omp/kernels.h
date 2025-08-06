/** Problem size along one side; total number of cells is this squared */
#define NUM 1024

// block size
#define BLOCK_SIZE 128

#define Real float
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f

/** SOR relaxation parameter */
const Real omega = 1.85f;

// begin of red_kernel
void red_kernel (const int numTeams,
                 const int numThreads,
                 const Real *__restrict__ aP,
                 const Real *__restrict__ aW,
                 const Real *__restrict__ aE,
                 const Real *__restrict__ aS,
                 const Real *__restrict__ aN,
                 const Real *__restrict__ b,
                 const Real *__restrict__ temp_black,
                       Real *__restrict__ temp_red,
                       Real *__restrict__ norm_L2)
{
  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(numTeams) num_threads(numThreads)
  for (int row = 1; row <= NUM/2; row++) {
    for (int col = 1; col <= NUM; col++) {
      int ind_red = col * ((NUM >> 1) + 2) + row;  					// local (red) index
      int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1);	// global index
  
      Real temp_old = temp_red[ind_red];
  
      Real res = b[ind] + (aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
            + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
            + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
            + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);
  
      Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
  
      temp_red[ind_red] = temp_new;
      res = temp_new - temp_old;
  
      norm_L2[ind_red] = res * res;
    }
  }
}
// end of red_kernel

void black_kernel (const int numTeams,
                   const int numThreads,
                   const Real *__restrict__ aP,
                   const Real *__restrict__ aW,
                   const Real *__restrict__ aE,
                   const Real *__restrict__ aS,
                   const Real *__restrict__ aN,
                   const Real *__restrict__ b,
                   const Real *__restrict__ temp_red,
                         Real *__restrict__ temp_black,
                         Real *__restrict__ norm_L2)
{
  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(numTeams) num_threads(numThreads)
  for (int row = 1; row <= NUM/2; row++) {
    for (int col = 1; col <= NUM; col++) {
      int ind_black = col * ((NUM >> 1) + 2) + row; // local (black) index
      int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1); // global index
  
      Real temp_old = temp_black[ind_black];
  
      Real res = b[ind] + (aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
            + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
            + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
            + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)]);
  
      Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
  
      temp_black[ind_black] = temp_new;
      res = temp_new - temp_old;
  
      norm_L2[ind_black] = res * res;
    }
  }
}
