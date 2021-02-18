/** GPU Laplace solver using optimized red-black Gauss–Seidel with SOR solver
 *
 * \author Kyle E. Niemeyer
 * \date 09/21/2012
 *
 * Solves Laplace's equation in 2D (e.g., heat conduction in a rectangular plate)
 * on GPU using CUDA with the red-black Gauss–Seidel with sucessive overrelaxation
 * (SOR) that has been "optimized". This means that the red and black kernels 
 * only loop over their respective cells, instead of over all cells and skipping
 * even/odd cells. This requires separate arrays for red and black cells.
 * 
 * Boundary conditions:
 * T = 0 at x = 0, x = L, y = 0
 * T = TN at y = H
 */

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "timer.h"

// CUDA libraries

/** Problem size along one side; total number of cells is this squared */
#define NUM 256

// block size
#define BLOCK_SIZE 128

#define Real float
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f

/** SOR relaxation parameter */
const Real omega = 1.85f;


///////////////////////////////////////////////////////////////////////////////

/** Function to evaluate coefficient matrix and right-hand side vector.
 * 
 * \param[in]		rowmax		number of rows
 * \param[in]		colmax		number of columns
 * \param[in]		th_cond		thermal conductivity
 * \param[in]		dx				grid size in x dimension (uniform)
 * \param[in]		dy				grid size in y dimension (uniform)
 * \param[in]		width			width of plate (z dimension)
 * \param[in]		TN				temperature at top boundary
 * \param[out]	aP				array of self coefficients
 * \param[out]	aW				array of west neighbor coefficients
 * \param[out]	aE				array of east neighbor coefficients
 * \param[out]	aS				array of south neighbor coefficients
 * \param[out]	aN				array of north neighbor coefficients
 * \param[out]	b					right-hand side array
 */
void fill_coeffs (int rowmax, int colmax, Real th_cond, Real dx, Real dy,
    Real width, Real TN, Real * aP, Real * aW, Real * aE, 
    Real * aS, Real * aN, Real * b)
{
  int col, row;
  for (col = 0; col < colmax; ++col) {
    for (row = 0; row < rowmax; ++row) {
      int ind = col * rowmax + row;

      b[ind] = ZERO;
      Real SP = ZERO;

      if (col == 0) {
        // left BC: temp = 0
        aW[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aW[ind] = th_cond * width * dy / dx;
      }

      if (col == colmax - 1) {
        // right BC: temp = 0
        aE[ind] = ZERO;
        SP = -TWO * th_cond * width * dy / dx;
      } else {
        aE[ind] = th_cond * width * dy / dx;
      }

      if (row == 0) {
        // bottom BC: temp = 0
        aS[ind] = ZERO;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aS[ind] = th_cond * width * dx / dy;
      }

      if (row == rowmax - 1) {
        // top BC: temp = TN
        aN[ind] = ZERO;
        b[ind] = TWO * th_cond * width * dx * TN / dy;
        SP = -TWO * th_cond * width * dx / dy;
      } else {
        aN[ind] = th_cond * width * dx / dy;
      }

      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
    } // end for row
  } // end for col
} // end fill_coeffs

///////////////////////////////////////////////////////////////////////////////

/** Function to update temperature for red cells
 * 
 * \param[in]			aP					array of self coefficients
 * \param[in]			aW					array of west neighbor coefficients
 * \param[in]			aE					array of east neighbor coefficients
 * \param[in]			aS					array of south neighbor coefficients
 * \param[in]			aN					array of north neighbor coefficients
 * \param[in]			b						right-hand side array
 * \param[in]			temp_black	temperatures of black cells, constant in this function
 * \param[inout]	temp_red		temperatures of red cells
 * \param[out]		bl_norm_L2	array with residual information for blocks
 */
void red_kernel (const Real * aP, const Real * aW, const Real * aE,
    const Real * aS, const Real * aN, const Real * b,
    const Real * temp_black, Real * temp_red,
    Real * norm_L2, sycl::nd_item<3> item_ct1)
{
    int row = 1 + (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
              item_ct1.get_local_id(2);
    int col = 1 + (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
              item_ct1.get_local_id(1);

  int ind_red = col * ((NUM >> 1) + 2) + row;  					// local (red) index
  int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1);	// global index

  Real temp_old = temp_red[ind_red];

  Real res = b[ind]
    + (aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
        + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
        + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
        + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);

  Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);

  temp_red[ind_red] = temp_new;
  res = temp_new - temp_old;

  norm_L2[ind_red] = res * res;

} // end red_kernel

///////////////////////////////////////////////////////////////////////////////

/** Function to update temperature for black cells
 * 
 * \param[in]			aP					array of self coefficients
 * \param[in]			aW					array of west neighbor coefficients
 * \param[in]			aE					array of east neighbor coefficients
 * \param[in]			aS					array of south neighbor coefficients
 * \param[in]			aN					array of north neighbor coefficients
 * \param[in]			b						right-hand side array
 * \param[in]			temp_red		temperatures of red cells, constant in this function
 * \param[inout]	temp_black	temperatures of black cells
 * \param[out]		bl_norm_L2	array with residual information for blocks
 */
void black_kernel (const Real * aP, const Real * aW, const Real * aE,
    const Real * aS, const Real * aN, const Real * b,
    const Real * temp_red, Real * temp_black, 
    Real * norm_L2, sycl::nd_item<3> item_ct1)
{
    int row = 1 + (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
              item_ct1.get_local_id(2);
    int col = 1 + (item_ct1.get_group(1) * item_ct1.get_local_range().get(1)) +
              item_ct1.get_local_id(1);

  int ind_black = col * ((NUM >> 1) + 2) + row;  				// local (black) index
  int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1);	// global index

  Real temp_old = temp_black[ind_black];

  Real res = b[ind]
    + (aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
        + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
        + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
        + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)]);

  Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);

  temp_black[ind_black] = temp_new;
  res = temp_new - temp_old;

  norm_L2[ind_black] = res * res;
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 * 
 * Contains iteration loop for red-black Gauss-Seidel with SOR GPU kernels
 */
int main(void) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

  // size of plate
  Real L = 1.0;
  Real H = 1.0;
  Real width = 0.01;

  // thermal conductivity
  Real th_cond = 1.0;

  // temperature at top boundary
  Real TN = 1.0;

  // SOR iteration tolerance
  Real tol = 1.e-6;

  // number of cells in x and y directions
  // including unused boundary cells
  int num_rows = (NUM / 2) + 2;
  int num_cols = NUM + 2;
  int size_temp = num_rows * num_cols;
  int size = NUM * NUM;

  // size of cells
  Real dx = L / NUM;
  Real dy = H / NUM;

  // iterations for Red-Black Gauss-Seidel with SOR
  int iter;
  int it_max = 1e6;

  // allocate memory
  Real *aP, *aW, *aE, *aS, *aN, *b;
  Real *temp_red, *temp_black;

  // arrays of coefficients
  aP = (Real *) calloc (size, sizeof(Real));
  aW = (Real *) calloc (size, sizeof(Real));
  aE = (Real *) calloc (size, sizeof(Real));
  aS = (Real *) calloc (size, sizeof(Real));
  aN = (Real *) calloc (size, sizeof(Real));

  // RHS
  b = (Real *) calloc (size, sizeof(Real));

  // temperature arrays
  temp_red = (Real *) calloc (size_temp, sizeof(Real));
  temp_black = (Real *) calloc (size_temp, sizeof(Real));

  // set coefficients
  fill_coeffs (NUM, NUM, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);

  int i;
  for (i = 0; i < size_temp; ++i) {
    temp_red[i] = ZERO;
    temp_black[i] = ZERO;
  }

  // block and grid dimensions
    sycl::range<3> dimBlock(BLOCK_SIZE, 1, 1);
    sycl::range<3> dimGrid(NUM / (2 * BLOCK_SIZE), NUM, 1);

  // residual
  Real *bl_norm_L2;

  // one for each temperature value
  int size_norm = size_temp;
  bl_norm_L2 = (Real *) calloc (size_norm, sizeof(Real));
  for (i = 0; i < size_norm; ++i) {
    bl_norm_L2[i] = ZERO;
  }

  // print problem info
  printf("Problem size: %d x %d \n", NUM, NUM);

  StartTimer();

  // allocate device memory
  Real *aP_d, *aW_d, *aE_d, *aS_d, *aN_d, *b_d;
  Real *temp_red_d;
  Real *temp_black_d;

    dpct::dpct_malloc((void **)&aP_d, size * sizeof(Real));
    dpct::dpct_malloc((void **)&aW_d, size * sizeof(Real));
    dpct::dpct_malloc((void **)&aE_d, size * sizeof(Real));
    dpct::dpct_malloc((void **)&aS_d, size * sizeof(Real));
    dpct::dpct_malloc((void **)&aN_d, size * sizeof(Real));
    dpct::dpct_malloc((void **)&b_d, size * sizeof(Real));
    dpct::dpct_malloc((void **)&temp_red_d, size_temp * sizeof(Real));
    dpct::dpct_malloc((void **)&temp_black_d, size_temp * sizeof(Real));

  // copy to device memory
    dpct::dpct_memcpy(aP_d, aP, size * sizeof(Real), dpct::host_to_device);
    dpct::dpct_memcpy(aW_d, aW, size * sizeof(Real), dpct::host_to_device);
    dpct::dpct_memcpy(aE_d, aE, size * sizeof(Real), dpct::host_to_device);
    dpct::dpct_memcpy(aS_d, aS, size * sizeof(Real), dpct::host_to_device);
    dpct::dpct_memcpy(aN_d, aN, size * sizeof(Real), dpct::host_to_device);
    dpct::dpct_memcpy(b_d, b, size * sizeof(Real), dpct::host_to_device);
    dpct::dpct_memcpy(temp_red_d, temp_red, size_temp * sizeof(Real),
                      dpct::host_to_device);
    dpct::dpct_memcpy(temp_black_d, temp_black, size_temp * sizeof(Real),
                      dpct::host_to_device);

  // residual
  Real *bl_norm_L2_d;
    dpct::dpct_malloc((void **)&bl_norm_L2_d, size_norm * sizeof(Real));
    dpct::dpct_memcpy(bl_norm_L2_d, bl_norm_L2, size_norm * sizeof(Real),
                      dpct::host_to_device);

  // iteration loop
  for (iter = 1; iter <= it_max; ++iter) {

    Real norm_L2 = ZERO;

        {
            dpct::buffer_t aP_d_buf_ct0 = dpct::get_buffer(aP_d);
            dpct::buffer_t aW_d_buf_ct1 = dpct::get_buffer(aW_d);
            dpct::buffer_t aE_d_buf_ct2 = dpct::get_buffer(aE_d);
            dpct::buffer_t aS_d_buf_ct3 = dpct::get_buffer(aS_d);
            dpct::buffer_t aN_d_buf_ct4 = dpct::get_buffer(aN_d);
            dpct::buffer_t b_d_buf_ct5 = dpct::get_buffer(b_d);
            dpct::buffer_t temp_black_d_buf_ct6 =
                dpct::get_buffer(temp_black_d);
            dpct::buffer_t temp_red_d_buf_ct7 = dpct::get_buffer(temp_red_d);
            dpct::buffer_t bl_norm_L2_d_buf_ct8 =
                dpct::get_buffer(bl_norm_L2_d);
            q_ct1.submit([&](sycl::handler &cgh) {
                auto aP_d_acc_ct0 =
                    aP_d_buf_ct0.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aW_d_acc_ct1 =
                    aW_d_buf_ct1.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aE_d_acc_ct2 =
                    aE_d_buf_ct2.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aS_d_acc_ct3 =
                    aS_d_buf_ct3.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aN_d_acc_ct4 =
                    aN_d_buf_ct4.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto b_d_acc_ct5 =
                    b_d_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
                auto temp_black_d_acc_ct6 =
                    temp_black_d_buf_ct6
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto temp_red_d_acc_ct7 =
                    temp_red_d_buf_ct7
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto bl_norm_L2_d_acc_ct8 =
                    bl_norm_L2_d_buf_ct8
                        .get_access<sycl::access::mode::read_write>(cgh);

                auto dpct_global_range = dimGrid * dimBlock;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(dimBlock.get(2),
                                                     dimBlock.get(1),
                                                     dimBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        red_kernel((const float *)(&aP_d_acc_ct0[0]),
                                   (const float *)(&aW_d_acc_ct1[0]),
                                   (const float *)(&aE_d_acc_ct2[0]),
                                   (const float *)(&aS_d_acc_ct3[0]),
                                   (const float *)(&aN_d_acc_ct4[0]),
                                   (const float *)(&b_d_acc_ct5[0]),
                                   (const float *)(&temp_black_d_acc_ct6[0]),
                                   (float *)(&temp_red_d_acc_ct7[0]),
                                   (float *)(&bl_norm_L2_d_acc_ct8[0]),
                                   item_ct1);
                    });
            });
        }

    // transfer residual value(s) back to CPU
        dpct::dpct_memcpy(bl_norm_L2, bl_norm_L2_d, size_norm * sizeof(Real),
                          dpct::device_to_host);

    // add red cell contributions to residual
    for (int i = 0; i < size_norm; ++i) {
      norm_L2 += bl_norm_L2[i];
    }

        {
            dpct::buffer_t aP_d_buf_ct0 = dpct::get_buffer(aP_d);
            dpct::buffer_t aW_d_buf_ct1 = dpct::get_buffer(aW_d);
            dpct::buffer_t aE_d_buf_ct2 = dpct::get_buffer(aE_d);
            dpct::buffer_t aS_d_buf_ct3 = dpct::get_buffer(aS_d);
            dpct::buffer_t aN_d_buf_ct4 = dpct::get_buffer(aN_d);
            dpct::buffer_t b_d_buf_ct5 = dpct::get_buffer(b_d);
            dpct::buffer_t temp_red_d_buf_ct6 = dpct::get_buffer(temp_red_d);
            dpct::buffer_t temp_black_d_buf_ct7 =
                dpct::get_buffer(temp_black_d);
            dpct::buffer_t bl_norm_L2_d_buf_ct8 =
                dpct::get_buffer(bl_norm_L2_d);
            q_ct1.submit([&](sycl::handler &cgh) {
                auto aP_d_acc_ct0 =
                    aP_d_buf_ct0.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aW_d_acc_ct1 =
                    aW_d_buf_ct1.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aE_d_acc_ct2 =
                    aE_d_buf_ct2.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aS_d_acc_ct3 =
                    aS_d_buf_ct3.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto aN_d_acc_ct4 =
                    aN_d_buf_ct4.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto b_d_acc_ct5 =
                    b_d_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
                auto temp_red_d_acc_ct6 =
                    temp_red_d_buf_ct6
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto temp_black_d_acc_ct7 =
                    temp_black_d_buf_ct7
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto bl_norm_L2_d_acc_ct8 =
                    bl_norm_L2_d_buf_ct8
                        .get_access<sycl::access::mode::read_write>(cgh);

                auto dpct_global_range = dimGrid * dimBlock;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(dimBlock.get(2),
                                                     dimBlock.get(1),
                                                     dimBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        black_kernel((const float *)(&aP_d_acc_ct0[0]),
                                     (const float *)(&aW_d_acc_ct1[0]),
                                     (const float *)(&aE_d_acc_ct2[0]),
                                     (const float *)(&aS_d_acc_ct3[0]),
                                     (const float *)(&aN_d_acc_ct4[0]),
                                     (const float *)(&b_d_acc_ct5[0]),
                                     (const float *)(&temp_red_d_acc_ct6[0]),
                                     (float *)(&temp_black_d_acc_ct7[0]),
                                     (float *)(&bl_norm_L2_d_acc_ct8[0]),
                                     item_ct1);
                    });
            });
        }

    // transfer residual value(s) back to CPU and 
    // add black cell contributions to residual
        dpct::dpct_memcpy(bl_norm_L2, bl_norm_L2_d, size_norm * sizeof(Real),
                          dpct::device_to_host);
    for (int i = 0; i < size_norm; ++i) {
      norm_L2 += bl_norm_L2[i];
    }

    // calculate residual
        norm_L2 = sqrt(norm_L2 / ((Real)size));

    if (iter % 1000 == 0) printf("%5d, %0.6f\n", iter, norm_L2);

    // if tolerance has been reached, end SOR iterations
    if (norm_L2 < tol) {
      break;
    }	
  }

  // transfer final temperature values back
    dpct::dpct_memcpy(temp_red, temp_red_d, size_temp * sizeof(Real),
                      dpct::device_to_host);
    dpct::dpct_memcpy(temp_black, temp_red_d, size_temp * sizeof(Real),
                      dpct::device_to_host);

  double runtime = GetTimer();

  printf("GPU\n");
  printf("Iterations: %i\n", iter);
  printf("Total time: %f s\n", runtime / 1000.0);

  // print temperature data to file
  FILE * pfile;
  pfile = fopen("temperature.dat", "w");

  if (pfile != NULL) {
    fprintf(pfile, "#x\ty\ttemp(K)\n");

    int row, col;
    for (row = 1; row < NUM + 1; ++row) {
      for (col = 1; col < NUM + 1; ++col) {
        Real x_pos = (col - 1) * dx + (dx / 2);
        Real y_pos = (row - 1) * dy + (dy / 2);

        if ((row + col) % 2 == 0) {
          // even, so red cell
          int ind = col * num_rows + (row + (col % 2)) / 2;
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_red[ind]);
        } else {
          // odd, so black cell
          int ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
          fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_black[ind]);
        }	
      }
      fprintf(pfile, "\n");
    }
  }
  fclose(pfile);

  // free device memory
    dpct::dpct_free(aP_d);
    dpct::dpct_free(aW_d);
    dpct::dpct_free(aE_d);
    dpct::dpct_free(aS_d);
    dpct::dpct_free(aN_d);
    dpct::dpct_free(b_d);
    dpct::dpct_free(temp_red_d);
    dpct::dpct_free(temp_black_d);
    dpct::dpct_free(bl_norm_L2_d);

  free(aP);
  free(aW);
  free(aE);
  free(aS);
  free(aN);
  free(b);
  free(temp_red);
  free(temp_black);
  free(bl_norm_L2);

  return 0;
}
