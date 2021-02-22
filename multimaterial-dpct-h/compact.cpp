#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <math.h>
#include <stdio.h>

char *cp_to_device(char *from, size_t size) {
  char *tmp;
  dpct::dpct_malloc((void **)&tmp, size);
  dpct::dpct_memcpy(tmp, from, size, dpct::host_to_device);
  return tmp;
}

void cp_to_host(char *to, char*from, size_t size) {
  dpct::dpct_memcpy(to, from, size, dpct::device_to_host);
  dpct::dpct_free(from);
}
void ccc_loop1(const int * __restrict imaterial, const int * __restrict nextfrac, const double * __restrict rho_compact, const double * __restrict rho_compact_list, 
    const double * __restrict  Vf_compact_list, const double * __restrict  V, double * __restrict rho_ave_compact, int sizex, int sizey, int * __restrict mmc_index,
    sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  int j = item_ct1.get_local_id(1) +
          item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
  if (i >= sizex || j >= sizey) return;
#ifdef FUSED
  double ave = 0.0;
  int ix = imaterial[i+sizex*j];

  if (ix <= 0) {
    // condition is 'ix >= 0', this is the equivalent of
    // 'until ix < 0' from the paper
#ifdef LINKED
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      ave += rho_compact_list[ix] * Vf_compact_list[ix];
    }
#else
    for (int idx = mmc_index[-ix]; idx < mmc_index[-ix+1]; idx++) {
      ave += rho_compact_list[idx] * Vf_compact_list[idx];  
    }
#endif
    rho_ave_compact[i+sizex*j] = ave/V[i+sizex*j];
  }
  else {
#endif
    // We use a distinct output array for averages.
    // In case of a pure cell, the average density equals to the total.
    rho_ave_compact[i+sizex*j] = rho_compact[i+sizex*j] / V[i+sizex*j];
#ifdef FUSED
  }
#endif
}


void ccc_loop1_2(const double * __restrict rho_compact_list, const double * __restrict  Vf_compact_list, const double * __restrict  V, double * __restrict rho_ave_compact, const int * __restrict mmc_index,  const int  mmc_cells,  const int * __restrict mmc_i,  const int * __restrict mmc_j, int sizex, int sizey,
                 sycl::nd_item<3> item_ct1) {
  int c = item_ct1.get_local_id(2) +
          item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  if (c >= mmc_cells) return;
  double ave = 0.0;
  for (int m = mmc_index[c]; m < mmc_index[c+1]; m++) {
    ave +=  rho_compact_list[m] * Vf_compact_list[m];
  }
  rho_ave_compact[mmc_i[c]+sizex*mmc_j[c]] = ave/V[mmc_i[c]+sizex*mmc_j[c]];
}

void ccc_loop2(const int * __restrict imaterial, const int * __restrict matids, const int * __restrict nextfrac, const double * __restrict rho_compact, const double * __restrict rho_compact_list, 
    const double * __restrict t_compact, const double * __restrict t_compact_list, 
    const double * __restrict  Vf_compact_list, const double * __restrict n, double * __restrict  p_compact, double * __restrict p_compact_list, int sizex, int sizey, int * __restrict mmc_index,
    sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  int j = item_ct1.get_local_id(1) +
          item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
  if (i >= sizex || j >= sizey) return;

  int ix = imaterial[i+sizex*j];
  if (ix <= 0) {
#ifdef FUSED
    // NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
    // The solution below is what I believe to good.

    // condition is 'ix >= 0', this is the equivalent of
    // 'until ix < 0' from the paper
#ifdef LINKED
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      double nm = n[matids[ix]];
      p_compact_list[ix] = (nm * rho_compact_list[ix] * t_compact_list[ix]) / Vf_compact_list[ix];
    }
#else
    for (int idx = mmc_index[-ix]; idx < mmc_index[-ix+1]; idx++) {
      double nm = n[matids[idx]];
      p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
    }
#endif
#endif
  }
  else {
    // NOTE: HACK: we index materials from zero, but zero can be a list index
    int mat = ix - 1;
    // NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
    p_compact[i+sizex*j] = n[mat] * rho_compact[i+sizex*j] * t_compact[i+sizex*j];;
  }
}
void ccc_loop2_2(const int * __restrict matids, const double * __restrict rho_compact_list, 
    const double * __restrict t_compact_list, const double * __restrict  Vf_compact_list, const double * __restrict n, double * __restrict p_compact_list, int * __restrict mmc_index, int mmc_cells,
    sycl::nd_item<3> item_ct1) {
  int idx = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  if (idx >= mmc_cells) return;
  double nm = n[matids[idx]];
  p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
}

void ccc_loop3(const int * __restrict imaterial, const int * __restrict nextfrac, const int * __restrict matids,
    const double * __restrict rho_compact, 
    const double * __restrict rho_compact_list, 
    double * __restrict rho_mat_ave_compact, 
    double * __restrict rho_mat_ave_compact_list, 
    const double * __restrict x, const double * __restrict y,
    int sizex, int sizey, int * __restrict mmc_index, sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  int j = item_ct1.get_local_id(1) +
          item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
  if (i >= sizex-1 || j >= sizey-1 || i < 1 || j < 1) return;


  // o: outer
  double xo = x[i+sizex*j];
  double yo = y[i+sizex*j];

  // There are at most 9 neighbours in 2D case.
  double dsqr[9];

  // for all neighbours
  for (int nj = -1; nj <= 1; nj++) {

    for (int ni = -1; ni <= 1; ni++) {

      dsqr[(nj+1)*3 + (ni+1)] = 0.0;

      // i: inner
      double xi = x[(i+ni)+sizex*(j+nj)];
      double yi = y[(i+ni)+sizex*(j+nj)];

      dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
      dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
    }
  }

  int ix = imaterial[i+sizex*j];

  if (ix <= 0) {
    // condition is 'ix >= 0', this is the equivalent of
    // 'until ix < 0' from the paper
#ifdef LINKED
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
#else
      for (int ix = mmc_index[-imaterial[i+sizex*j]]; ix < mmc_index[-imaterial[i+sizex*j]+1]; ix++) {
#endif
        int mat = matids[ix];
        double rho_sum = 0.0;
        int Nn = 0;

        // for all neighbours
        for (int nj = -1; nj <= 1; nj++) {

          for (int ni = -1; ni <= 1; ni++) {

            int ci = i+ni, cj = j+nj;
            int jx = imaterial[ci+sizex*cj];

            if (jx <= 0) {
              // condition is 'jx >= 0', this is the equivalent of
              // 'until jx < 0' from the paper
#ifdef LINKED
              for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
#else
                for (int jx = mmc_index[-imaterial[ci+sizex*cj]]; jx < mmc_index[-imaterial[ci+sizex*cj]+1]; jx++) {
#endif
                  if (matids[jx] == mat) {
                    rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
                    Nn += 1;

                    // The loop has an extra condition: "and not found".
                    // This makes sense, if the material is found, there won't be any more of the same.
                    break;
                  }
                }
              }
              else {
                // NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
                // In contrast, Algorithm 10 loads matids[jx] which I think is wrong.

                // NOTE: HACK: we index materials from zero, but zero can be a list index
                int mat_neighbour = jx - 1;
                if (mat == mat_neighbour) {
                  rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
                  Nn += 1;
                }
              } // end if (jx <= 0)
            } // end for (int ni)
          } // end for (int nj)

          rho_mat_ave_compact_list[ix] = rho_sum / Nn;
        } // end for (ix = -ix)
      } // end if (ix <= 0)
      else {
        // NOTE: In this case, the cell is a pure cell, its material index is in ix.
        // In contrast, Algorithm 10 loads matids[ix] which I think is wrong.

        // NOTE: HACK: we index materials from zero, but zero can be a list index
        int mat = ix - 1;

        double rho_sum = 0.0;
        int Nn = 0;

        // for all neighbours
        for (int nj = -1; nj <= 1; nj++) {
          if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
            continue;

          for (int ni = -1; ni <= 1; ni++) {
            if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
              continue;

            int ci = i+ni, cj = j+nj;
            int jx = imaterial[ci+sizex*cj];

            if (jx <= 0) {
              // condition is 'jx >= 0', this is the equivalent of
              // 'until jx < 0' from the paper
#ifdef LINKED
              for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
#else
                for (int jx = mmc_index[-imaterial[ci+sizex*cj]]; jx < mmc_index[-imaterial[ci+sizex*cj]+1]; jx++) {
#endif
                  if (matids[jx] == mat) {
                    rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
                    Nn += 1;

                    // The loop has an extra condition: "and not found".
                    // This makes sense, if the material is found, there won't be any more of the same.
                    break;
                  }
                }
              }
              else {
                // NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
                // In contrast, Algorithm 10 loads matids[jx] which I think is wrong.

                // NOTE: HACK: we index materials from zero, but zero can be a list index
                int mat_neighbour = jx - 1;
                if (mat == mat_neighbour) {
                  rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
                  Nn += 1;
                }
              } // end if (jx <= 0)
            } // end for (int ni)
          } // end for (int nj)

          rho_mat_ave_compact[i+sizex*j] = rho_sum / Nn;
        } // end else

      }


      struct full_data
      {
        int sizex;
        int sizey;
        int Nmats;
        double * __restrict__ rho;
        double * __restrict__ rho_mat_ave;
        double * __restrict__ p;
        double * __restrict__ Vf;
        double * __restrict__ t;
        double * __restrict__ V;
        double * __restrict__ x;
        double * __restrict__ y;
        double * __restrict__ n;
        double * __restrict__ rho_ave;
      };

      struct compact_data
      {
        int sizex;
        int sizey;
        int Nmats;
        double * __restrict__ rho_compact;
        double * __restrict__ rho_compact_list;
        double * __restrict__ rho_mat_ave_compact;
        double * __restrict__ rho_mat_ave_compact_list;
        double * __restrict__ p_compact;
        double * __restrict__ p_compact_list;
        double * __restrict__ Vf_compact_list;
        double * __restrict__ t_compact;
        double * __restrict__ t_compact_list;
        double * __restrict__ V;
        double * __restrict__ x;
        double * __restrict__ y;
        double * __restrict__ n;
        double * __restrict__ rho_ave_compact;
        int * __restrict__ imaterial;
        int * __restrict__ matids;
        int * __restrict__ nextfrac;
        int * __restrict__ mmc_index;
        int * __restrict__ mmc_i;
        int * __restrict__ mmc_j;
        int mm_len;
        int mmc_cells;
      };


      void compact_cell_centric(full_data cc, compact_data ccc, double &a1, double &a2, double &a3, int argc, char** argv)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

        int sizex = cc.sizex;
        int sizey = cc.sizey;
        int Nmats = cc.Nmats;
        int mmc_cells = ccc.mmc_cells;
        int mm_len = ccc.mm_len;

        int    *d_imaterial = (int *)cp_to_device((char*)ccc.imaterial, sizex*sizey*sizeof(int));
        int    *d_matids = (int *)cp_to_device((char*)ccc.matids, mm_len*sizeof(int));
        int    *d_nextfrac = (int *)cp_to_device((char*)ccc.nextfrac, mm_len*sizeof(int));
        int    *d_mmc_index = (int *)cp_to_device((char*)ccc.mmc_index, (mmc_cells+1)*sizeof(int));
        int    *d_mmc_i = (int *)cp_to_device((char*)ccc.mmc_i, (mmc_cells)*sizeof(int));
        int    *d_mmc_j = (int *)cp_to_device((char*)ccc.mmc_j, (mmc_cells)*sizeof(int));
        double *d_x = (double *)cp_to_device((char*)ccc.x, sizex*sizey*sizeof(double));
        double *d_y = (double *)cp_to_device((char*)ccc.y, sizex*sizey*sizeof(double));
        double *d_rho_compact = (double *)cp_to_device((char*)ccc.rho_compact, sizex*sizey*sizeof(double));
        double *d_rho_compact_list = (double *)cp_to_device((char*)ccc.rho_compact_list,mm_len*sizeof(double));
        double *d_rho_mat_ave_compact = (double *)cp_to_device((char*)ccc.rho_mat_ave_compact, sizex*sizey*sizeof(double));
        double *d_rho_mat_ave_compact_list = (double *)cp_to_device((char*)ccc.rho_mat_ave_compact_list,mm_len*sizeof(double));
        double *d_p_compact = (double *)cp_to_device((char*)ccc.p_compact, sizex*sizey*sizeof(double));
        double *d_p_compact_list = (double *)cp_to_device((char*)ccc.p_compact_list,mm_len*sizeof(double));
        double *d_t_compact = (double *)cp_to_device((char*)ccc.t_compact, sizex*sizey*sizeof(double));
        double *d_t_compact_list = (double *)cp_to_device((char*)ccc.t_compact_list,mm_len*sizeof(double));
        double *d_Vf_compact_list = (double *)cp_to_device((char*)ccc.Vf_compact_list, mm_len*sizeof(double));
        double *d_V = (double *)cp_to_device((char*)ccc.V, sizex*sizey*sizeof(double));
        double *d_n = (double *)cp_to_device((char*)ccc.n, Nmats*sizeof(double));
        double *d_rho_ave_compact = (double *)cp_to_device((char*)ccc.rho_ave_compact, sizex*sizey*sizeof(double));

        int thx = 32;
        int thy = 4;
  sycl::range<3> threads(thx, thy, 1);
  sycl::range<3> blocks((sizex - 1) / thx + 1, (sizey - 1) / thy + 1, 1);

        // Cell-centric algorithms
        // Computational loop 1 - average density in cell
#ifdef DEBUG
  dev_ct1.queues_wait_and_throw();
        auto t0 = std::chrono::system_clock::now();
#endif
  {
    std::pair<dpct::buffer_t, size_t> d_imaterial_buf_ct0 =
        dpct::get_buffer_and_offset(d_imaterial);
    size_t d_imaterial_offset_ct0 = d_imaterial_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_nextfrac_buf_ct1 =
        dpct::get_buffer_and_offset(d_nextfrac);
    size_t d_nextfrac_offset_ct1 = d_nextfrac_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_rho_compact_buf_ct2 =
        dpct::get_buffer_and_offset(d_rho_compact);
    size_t d_rho_compact_offset_ct2 = d_rho_compact_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_rho_compact_list_buf_ct3 =
        dpct::get_buffer_and_offset(d_rho_compact_list);
    size_t d_rho_compact_list_offset_ct3 = d_rho_compact_list_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> d_Vf_compact_list_buf_ct4 =
        dpct::get_buffer_and_offset(d_Vf_compact_list);
    size_t d_Vf_compact_list_offset_ct4 = d_Vf_compact_list_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_V_buf_ct5 =
        dpct::get_buffer_and_offset(d_V);
    size_t d_V_offset_ct5 = d_V_buf_ct5.second;
    std::pair<dpct::buffer_t, size_t> d_rho_ave_compact_buf_ct6 =
        dpct::get_buffer_and_offset(d_rho_ave_compact);
    size_t d_rho_ave_compact_offset_ct6 = d_rho_ave_compact_buf_ct6.second;
    std::pair<dpct::buffer_t, size_t> d_mmc_index_buf_ct9 =
        dpct::get_buffer_and_offset(d_mmc_index);
    size_t d_mmc_index_offset_ct9 = d_mmc_index_buf_ct9.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_imaterial_acc_ct0 =
          d_imaterial_buf_ct0.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_nextfrac_acc_ct1 =
          d_nextfrac_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_rho_compact_acc_ct2 =
          d_rho_compact_buf_ct2.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_rho_compact_list_acc_ct3 =
          d_rho_compact_list_buf_ct3.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_Vf_compact_list_acc_ct4 =
          d_Vf_compact_list_buf_ct4.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_V_acc_ct5 =
          d_V_buf_ct5.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_rho_ave_compact_acc_ct6 =
          d_rho_ave_compact_buf_ct6.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_mmc_index_acc_ct9 =
          d_mmc_index_buf_ct9.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = sycl::range<3>(blocks) * sycl::range<3>(threads);
      auto dpct_local_range = sycl::range<3>(threads);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1),
                             dpct_local_range.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            const int *d_imaterial_ct0 =
                (const int *)(&d_imaterial_acc_ct0[0] + d_imaterial_offset_ct0);
            const int *d_nextfrac_ct1 =
                (const int *)(&d_nextfrac_acc_ct1[0] + d_nextfrac_offset_ct1);
            const double *d_rho_compact_ct2 =
                (const double *)(&d_rho_compact_acc_ct2[0] +
                                 d_rho_compact_offset_ct2);
            const double *d_rho_compact_list_ct3 =
                (const double *)(&d_rho_compact_list_acc_ct3[0] +
                                 d_rho_compact_list_offset_ct3);
            const double *d_Vf_compact_list_ct4 =
                (const double *)(&d_Vf_compact_list_acc_ct4[0] +
                                 d_Vf_compact_list_offset_ct4);
            const double *d_V_ct5 =
                (const double *)(&d_V_acc_ct5[0] + d_V_offset_ct5);
            double *d_rho_ave_compact_ct6 =
                (double *)(&d_rho_ave_compact_acc_ct6[0] +
                           d_rho_ave_compact_offset_ct6);
            int *d_mmc_index_ct9 =
                (int *)(&d_mmc_index_acc_ct9[0] + d_mmc_index_offset_ct9);
            ccc_loop1(d_imaterial_ct0, d_nextfrac_ct1, d_rho_compact_ct2,
                      d_rho_compact_list_ct3, d_Vf_compact_list_ct4, d_V_ct5,
                      d_rho_ave_compact_ct6, sizex, sizey, d_mmc_index_ct9,
                      item_ct1);
          });
    });
  }
#ifndef FUSED
  {
    std::pair<dpct::buffer_t, size_t> d_rho_compact_list_buf_ct0 =
        dpct::get_buffer_and_offset(d_rho_compact_list);
    size_t d_rho_compact_list_offset_ct0 = d_rho_compact_list_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_Vf_compact_list_buf_ct1 =
        dpct::get_buffer_and_offset(d_Vf_compact_list);
    size_t d_Vf_compact_list_offset_ct1 = d_Vf_compact_list_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_V_buf_ct2 =
        dpct::get_buffer_and_offset(d_V);
    size_t d_V_offset_ct2 = d_V_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_rho_ave_compact_buf_ct3 =
        dpct::get_buffer_and_offset(d_rho_ave_compact);
    size_t d_rho_ave_compact_offset_ct3 = d_rho_ave_compact_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> d_mmc_index_buf_ct4 =
        dpct::get_buffer_and_offset(d_mmc_index);
    size_t d_mmc_index_offset_ct4 = d_mmc_index_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_mmc_i_buf_ct6 =
        dpct::get_buffer_and_offset(d_mmc_i);
    size_t d_mmc_i_offset_ct6 = d_mmc_i_buf_ct6.second;
    std::pair<dpct::buffer_t, size_t> d_mmc_j_buf_ct7 =
        dpct::get_buffer_and_offset(d_mmc_j);
    size_t d_mmc_j_offset_ct7 = d_mmc_j_buf_ct7.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_rho_compact_list_acc_ct0 =
          d_rho_compact_list_buf_ct0.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_Vf_compact_list_acc_ct1 =
          d_Vf_compact_list_buf_ct1.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_V_acc_ct2 =
          d_V_buf_ct2.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_rho_ave_compact_acc_ct3 =
          d_rho_ave_compact_buf_ct3.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_mmc_index_acc_ct4 =
          d_mmc_index_buf_ct4.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_mmc_i_acc_ct6 =
          d_mmc_i_buf_ct6.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_mmc_j_acc_ct7 =
          d_mmc_j_buf_ct7.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, (mmc_cells - 1) / (thx * thy) + 1) *
                  sycl::range<3>(1, 1, (thx * thy)),
              sycl::range<3>(1, 1, (thx * thy))),
          [=](sycl::nd_item<3> item_ct1) {
            const double *d_rho_compact_list_ct0 =
                (const double *)(&d_rho_compact_list_acc_ct0[0] +
                                 d_rho_compact_list_offset_ct0);
            const double *d_Vf_compact_list_ct1 =
                (const double *)(&d_Vf_compact_list_acc_ct1[0] +
                                 d_Vf_compact_list_offset_ct1);
            const double *d_V_ct2 =
                (const double *)(&d_V_acc_ct2[0] + d_V_offset_ct2);
            double *d_rho_ave_compact_ct3 =
                (double *)(&d_rho_ave_compact_acc_ct3[0] +
                           d_rho_ave_compact_offset_ct3);
            const int *d_mmc_index_ct4 =
                (const int *)(&d_mmc_index_acc_ct4[0] + d_mmc_index_offset_ct4);
            const int *d_mmc_i_ct6 =
                (const int *)(&d_mmc_i_acc_ct6[0] + d_mmc_i_offset_ct6);
            const int *d_mmc_j_ct7 =
                (const int *)(&d_mmc_j_acc_ct7[0] + d_mmc_j_offset_ct7);
            ccc_loop1_2(d_rho_compact_list_ct0, d_Vf_compact_list_ct1, d_V_ct2,
                        d_rho_ave_compact_ct3, d_mmc_index_ct4, mmc_cells,
                        d_mmc_i_ct6, d_mmc_j_ct7, sizex, sizey, item_ct1);
          });
    });
  }
#endif
#ifdef DEBUG
  dev_ct1.queues_wait_and_throw();
        std::chrono::duration<double> t1 = std::chrono::system_clock::now() - t0;
        printf("Compact matrix, cell centric, alg 1: %g sec\n", t1.count());
        // Computational loop 2 - Pressure for each cell and each material
        t0 = std::chrono::system_clock::now();
#endif
  {
    std::pair<dpct::buffer_t, size_t> d_imaterial_buf_ct0 =
        dpct::get_buffer_and_offset(d_imaterial);
    size_t d_imaterial_offset_ct0 = d_imaterial_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_matids_buf_ct1 =
        dpct::get_buffer_and_offset(d_matids);
    size_t d_matids_offset_ct1 = d_matids_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_nextfrac_buf_ct2 =
        dpct::get_buffer_and_offset(d_nextfrac);
    size_t d_nextfrac_offset_ct2 = d_nextfrac_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_rho_compact_buf_ct3 =
        dpct::get_buffer_and_offset(d_rho_compact);
    size_t d_rho_compact_offset_ct3 = d_rho_compact_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> d_rho_compact_list_buf_ct4 =
        dpct::get_buffer_and_offset(d_rho_compact_list);
    size_t d_rho_compact_list_offset_ct4 = d_rho_compact_list_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_t_compact_buf_ct5 =
        dpct::get_buffer_and_offset(d_t_compact);
    size_t d_t_compact_offset_ct5 = d_t_compact_buf_ct5.second;
    std::pair<dpct::buffer_t, size_t> d_t_compact_list_buf_ct6 =
        dpct::get_buffer_and_offset(d_t_compact_list);
    size_t d_t_compact_list_offset_ct6 = d_t_compact_list_buf_ct6.second;
    std::pair<dpct::buffer_t, size_t> d_Vf_compact_list_buf_ct7 =
        dpct::get_buffer_and_offset(d_Vf_compact_list);
    size_t d_Vf_compact_list_offset_ct7 = d_Vf_compact_list_buf_ct7.second;
    std::pair<dpct::buffer_t, size_t> d_n_buf_ct8 =
        dpct::get_buffer_and_offset(d_n);
    size_t d_n_offset_ct8 = d_n_buf_ct8.second;
    std::pair<dpct::buffer_t, size_t> d_p_compact_buf_ct9 =
        dpct::get_buffer_and_offset(d_p_compact);
    size_t d_p_compact_offset_ct9 = d_p_compact_buf_ct9.second;
    std::pair<dpct::buffer_t, size_t> d_p_compact_list_buf_ct10 =
        dpct::get_buffer_and_offset(d_p_compact_list);
    size_t d_p_compact_list_offset_ct10 = d_p_compact_list_buf_ct10.second;
    std::pair<dpct::buffer_t, size_t> d_mmc_index_buf_ct13 =
        dpct::get_buffer_and_offset(d_mmc_index);
    size_t d_mmc_index_offset_ct13 = d_mmc_index_buf_ct13.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_imaterial_acc_ct0 =
          d_imaterial_buf_ct0.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_matids_acc_ct1 =
          d_matids_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_nextfrac_acc_ct2 =
          d_nextfrac_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_rho_compact_acc_ct3 =
          d_rho_compact_buf_ct3.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_rho_compact_list_acc_ct4 =
          d_rho_compact_list_buf_ct4.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_t_compact_acc_ct5 =
          d_t_compact_buf_ct5.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_t_compact_list_acc_ct6 =
          d_t_compact_list_buf_ct6.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_Vf_compact_list_acc_ct7 =
          d_Vf_compact_list_buf_ct7.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_n_acc_ct8 =
          d_n_buf_ct8.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_p_compact_acc_ct9 =
          d_p_compact_buf_ct9.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_p_compact_list_acc_ct10 =
          d_p_compact_list_buf_ct10.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_mmc_index_acc_ct13 =
          d_mmc_index_buf_ct13.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = sycl::range<3>(blocks) * sycl::range<3>(threads);
      auto dpct_local_range = sycl::range<3>(threads);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1),
                             dpct_local_range.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            const int *d_imaterial_ct0 =
                (const int *)(&d_imaterial_acc_ct0[0] + d_imaterial_offset_ct0);
            const int *d_matids_ct1 =
                (const int *)(&d_matids_acc_ct1[0] + d_matids_offset_ct1);
            const int *d_nextfrac_ct2 =
                (const int *)(&d_nextfrac_acc_ct2[0] + d_nextfrac_offset_ct2);
            const double *d_rho_compact_ct3 =
                (const double *)(&d_rho_compact_acc_ct3[0] +
                                 d_rho_compact_offset_ct3);
            const double *d_rho_compact_list_ct4 =
                (const double *)(&d_rho_compact_list_acc_ct4[0] +
                                 d_rho_compact_list_offset_ct4);
            const double *d_t_compact_ct5 =
                (const double *)(&d_t_compact_acc_ct5[0] +
                                 d_t_compact_offset_ct5);
            const double *d_t_compact_list_ct6 =
                (const double *)(&d_t_compact_list_acc_ct6[0] +
                                 d_t_compact_list_offset_ct6);
            const double *d_Vf_compact_list_ct7 =
                (const double *)(&d_Vf_compact_list_acc_ct7[0] +
                                 d_Vf_compact_list_offset_ct7);
            const double *d_n_ct8 =
                (const double *)(&d_n_acc_ct8[0] + d_n_offset_ct8);
            double *d_p_compact_ct9 =
                (double *)(&d_p_compact_acc_ct9[0] + d_p_compact_offset_ct9);
            double *d_p_compact_list_ct10 =
                (double *)(&d_p_compact_list_acc_ct10[0] +
                           d_p_compact_list_offset_ct10);
            int *d_mmc_index_ct13 =
                (int *)(&d_mmc_index_acc_ct13[0] + d_mmc_index_offset_ct13);
            ccc_loop2(d_imaterial_ct0, d_matids_ct1, d_nextfrac_ct2,
                      d_rho_compact_ct3, d_rho_compact_list_ct4,
                      d_t_compact_ct5, d_t_compact_list_ct6,
                      d_Vf_compact_list_ct7, d_n_ct8, d_p_compact_ct9,
                      d_p_compact_list_ct10, sizex, sizey, d_mmc_index_ct13,
                      item_ct1);
          });
    });
  }
#ifndef FUSED
  {
    std::pair<dpct::buffer_t, size_t> d_matids_buf_ct0 =
        dpct::get_buffer_and_offset(d_matids);
    size_t d_matids_offset_ct0 = d_matids_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_rho_compact_list_buf_ct1 =
        dpct::get_buffer_and_offset(d_rho_compact_list);
    size_t d_rho_compact_list_offset_ct1 = d_rho_compact_list_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_t_compact_list_buf_ct2 =
        dpct::get_buffer_and_offset(d_t_compact_list);
    size_t d_t_compact_list_offset_ct2 = d_t_compact_list_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_Vf_compact_list_buf_ct3 =
        dpct::get_buffer_and_offset(d_Vf_compact_list);
    size_t d_Vf_compact_list_offset_ct3 = d_Vf_compact_list_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> d_n_buf_ct4 =
        dpct::get_buffer_and_offset(d_n);
    size_t d_n_offset_ct4 = d_n_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_p_compact_list_buf_ct5 =
        dpct::get_buffer_and_offset(d_p_compact_list);
    size_t d_p_compact_list_offset_ct5 = d_p_compact_list_buf_ct5.second;
    std::pair<dpct::buffer_t, size_t> d_mmc_index_buf_ct6 =
        dpct::get_buffer_and_offset(d_mmc_index);
    size_t d_mmc_index_offset_ct6 = d_mmc_index_buf_ct6.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_matids_acc_ct0 =
          d_matids_buf_ct0.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_rho_compact_list_acc_ct1 =
          d_rho_compact_list_buf_ct1.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_t_compact_list_acc_ct2 =
          d_t_compact_list_buf_ct2.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_Vf_compact_list_acc_ct3 =
          d_Vf_compact_list_buf_ct3.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_n_acc_ct4 =
          d_n_buf_ct4.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_p_compact_list_acc_ct5 =
          d_p_compact_list_buf_ct5.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_mmc_index_acc_ct6 =
          d_mmc_index_buf_ct6.first.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, (mm_len - 1) / (thx * thy) + 1) *
                  sycl::range<3>(1, 1, (thx * thy)),
              sycl::range<3>(1, 1, (thx * thy))),
          [=](sycl::nd_item<3> item_ct1) {
            const int *d_matids_ct0 =
                (const int *)(&d_matids_acc_ct0[0] + d_matids_offset_ct0);
            const double *d_rho_compact_list_ct1 =
                (const double *)(&d_rho_compact_list_acc_ct1[0] +
                                 d_rho_compact_list_offset_ct1);
            const double *d_t_compact_list_ct2 =
                (const double *)(&d_t_compact_list_acc_ct2[0] +
                                 d_t_compact_list_offset_ct2);
            const double *d_Vf_compact_list_ct3 =
                (const double *)(&d_Vf_compact_list_acc_ct3[0] +
                                 d_Vf_compact_list_offset_ct3);
            const double *d_n_ct4 =
                (const double *)(&d_n_acc_ct4[0] + d_n_offset_ct4);
            double *d_p_compact_list_ct5 =
                (double *)(&d_p_compact_list_acc_ct5[0] +
                           d_p_compact_list_offset_ct5);
            int *d_mmc_index_ct6 =
                (int *)(&d_mmc_index_acc_ct6[0] + d_mmc_index_offset_ct6);
            ccc_loop2_2(d_matids_ct0, d_rho_compact_list_ct1,
                        d_t_compact_list_ct2, d_Vf_compact_list_ct3, d_n_ct4,
                        d_p_compact_list_ct5, d_mmc_index_ct6, mm_len,
                        item_ct1);
          });
    });
  }
#endif
#ifdef DEBUG
  dev_ct1.queues_wait_and_throw();
        std::chrono::duration<double> t2 = std::chrono::system_clock::now() - t0;
        printf("Compact matrix, cell centric, alg 2: %g sec\n", t2.count());
        // Computational loop 3 - Average density of each material over neighborhood of each cell
        t0 = std::chrono::system_clock::now();
#endif
  {
    std::pair<dpct::buffer_t, size_t> d_imaterial_buf_ct0 =
        dpct::get_buffer_and_offset(d_imaterial);
    size_t d_imaterial_offset_ct0 = d_imaterial_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_nextfrac_buf_ct1 =
        dpct::get_buffer_and_offset(d_nextfrac);
    size_t d_nextfrac_offset_ct1 = d_nextfrac_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_matids_buf_ct2 =
        dpct::get_buffer_and_offset(d_matids);
    size_t d_matids_offset_ct2 = d_matids_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_rho_compact_buf_ct3 =
        dpct::get_buffer_and_offset(d_rho_compact);
    size_t d_rho_compact_offset_ct3 = d_rho_compact_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> d_rho_compact_list_buf_ct4 =
        dpct::get_buffer_and_offset(d_rho_compact_list);
    size_t d_rho_compact_list_offset_ct4 = d_rho_compact_list_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_rho_mat_ave_compact_buf_ct5 =
        dpct::get_buffer_and_offset(d_rho_mat_ave_compact);
    size_t d_rho_mat_ave_compact_offset_ct5 =
        d_rho_mat_ave_compact_buf_ct5.second;
    std::pair<dpct::buffer_t, size_t> d_rho_mat_ave_compact_list_buf_ct6 =
        dpct::get_buffer_and_offset(d_rho_mat_ave_compact_list);
    size_t d_rho_mat_ave_compact_list_offset_ct6 =
        d_rho_mat_ave_compact_list_buf_ct6.second;
    std::pair<dpct::buffer_t, size_t> d_x_buf_ct7 =
        dpct::get_buffer_and_offset(d_x);
    size_t d_x_offset_ct7 = d_x_buf_ct7.second;
    std::pair<dpct::buffer_t, size_t> d_y_buf_ct8 =
        dpct::get_buffer_and_offset(d_y);
    size_t d_y_offset_ct8 = d_y_buf_ct8.second;
    std::pair<dpct::buffer_t, size_t> d_mmc_index_buf_ct11 =
        dpct::get_buffer_and_offset(d_mmc_index);
    size_t d_mmc_index_offset_ct11 = d_mmc_index_buf_ct11.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_imaterial_acc_ct0 =
          d_imaterial_buf_ct0.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_nextfrac_acc_ct1 =
          d_nextfrac_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_matids_acc_ct2 =
          d_matids_buf_ct2.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_rho_compact_acc_ct3 =
          d_rho_compact_buf_ct3.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_rho_compact_list_acc_ct4 =
          d_rho_compact_list_buf_ct4.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_rho_mat_ave_compact_acc_ct5 =
          d_rho_mat_ave_compact_buf_ct5.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_rho_mat_ave_compact_list_acc_ct6 =
          d_rho_mat_ave_compact_list_buf_ct6.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_x_acc_ct7 =
          d_x_buf_ct7.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_y_acc_ct8 =
          d_y_buf_ct8.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_mmc_index_acc_ct11 =
          d_mmc_index_buf_ct11.first.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = sycl::range<3>(blocks) * sycl::range<3>(threads);
      auto dpct_local_range = sycl::range<3>(threads);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1),
                             dpct_local_range.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            const int *d_imaterial_ct0 =
                (const int *)(&d_imaterial_acc_ct0[0] + d_imaterial_offset_ct0);
            const int *d_nextfrac_ct1 =
                (const int *)(&d_nextfrac_acc_ct1[0] + d_nextfrac_offset_ct1);
            const int *d_matids_ct2 =
                (const int *)(&d_matids_acc_ct2[0] + d_matids_offset_ct2);
            const double *d_rho_compact_ct3 =
                (const double *)(&d_rho_compact_acc_ct3[0] +
                                 d_rho_compact_offset_ct3);
            const double *d_rho_compact_list_ct4 =
                (const double *)(&d_rho_compact_list_acc_ct4[0] +
                                 d_rho_compact_list_offset_ct4);
            double *d_rho_mat_ave_compact_ct5 =
                (double *)(&d_rho_mat_ave_compact_acc_ct5[0] +
                           d_rho_mat_ave_compact_offset_ct5);
            double *d_rho_mat_ave_compact_list_ct6 =
                (double *)(&d_rho_mat_ave_compact_list_acc_ct6[0] +
                           d_rho_mat_ave_compact_list_offset_ct6);
            const double *d_x_ct7 =
                (const double *)(&d_x_acc_ct7[0] + d_x_offset_ct7);
            const double *d_y_ct8 =
                (const double *)(&d_y_acc_ct8[0] + d_y_offset_ct8);
            int *d_mmc_index_ct11 =
                (int *)(&d_mmc_index_acc_ct11[0] + d_mmc_index_offset_ct11);
            ccc_loop3(d_imaterial_ct0, d_nextfrac_ct1, d_matids_ct2,
                      d_rho_compact_ct3, d_rho_compact_list_ct4,
                      d_rho_mat_ave_compact_ct5, d_rho_mat_ave_compact_list_ct6,
                      d_x_ct7, d_y_ct8, sizex, sizey, d_mmc_index_ct11,
                      item_ct1);
          });
    });
  }
#ifdef DEBUG
  dev_ct1.queues_wait_and_throw();
        std::chrono::duration<double> t3 = std::chrono::system_clock::now() - t0;
        printf("Compact matrix, cell centric, alg 3: %g sec\n", t3.count());
#endif

        cp_to_host((char*)ccc.x, (char*)d_x, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.y, (char*)d_y, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.rho_compact, (char*)d_rho_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.rho_compact_list, (char*)d_rho_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.rho_mat_ave_compact, (char*)d_rho_mat_ave_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.rho_mat_ave_compact_list, (char*)d_rho_mat_ave_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.p_compact, (char*)d_p_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.p_compact_list, (char*)d_p_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.t_compact, (char*)d_t_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.t_compact_list, (char*)d_t_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.Vf_compact_list, (char*)d_Vf_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.V, (char*)d_V, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.n, (char*)d_n, Nmats*sizeof(double));
        cp_to_host((char*)ccc.rho_ave_compact, (char*)d_rho_ave_compact, sizex*sizey*sizeof(double));

        // cudaFree are included in cp_to_host
      }

      bool compact_check_results(full_data cc, compact_data ccc)
      {
        int sizex = cc.sizex;
        int sizey = cc.sizey;
        int Nmats = cc.Nmats;
        //int mmc_cells = ccc.mmc_cells;
        //int mm_len = ccc.mm_len;


        printf("Checking results of compact representation... ");

        for (int j = 0; j < sizey; j++) {
          for (int i = 0; i < sizex; i++) {
      if (fabs(cc.rho_ave[i + sizex * j] - ccc.rho_ave_compact[i + sizex * j]) >
          0.0001) {
              printf("1. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d)\n",
                  cc.rho_ave[i+sizex*j], ccc.rho_ave_compact[i+sizex*j], i, j);
              return false;
            }
            int ix = ccc.imaterial[i+sizex*j];
            if (ix <= 0) {
#ifdef LINKED
              for (ix = -ix; ix >= 0; ix = ccc.nextfrac[ix]) {
#else
                for (int ix = ccc.mmc_index[-ccc.imaterial[i+sizex*j]]; ix < ccc.mmc_index[-ccc.imaterial[i+sizex*j]+1]; ix++) {
#endif
                  int mat = ccc.matids[ix];
          if (fabs(cc.p[(i + sizex * j) * Nmats + mat] -
                   ccc.p_compact_list[ix]) > 0.0001) {
                    printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                        cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact_list[ix], i, j, mat);
                    return false;
                  }

          if (fabs(cc.rho_mat_ave[(i + sizex * j) * Nmats + mat] -
                   ccc.rho_mat_ave_compact_list[ix]) > 0.0001) {
                    printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                        cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], ccc.rho_mat_ave_compact_list[ix], i, j, mat);
                    return false;
                  }
                }
              }
              else {
                // NOTE: HACK: we index materials from zero, but zero can be a list index
                int mat = ix - 1;
        if (fabs(cc.p[(i + sizex * j) * Nmats + mat] -
                 ccc.p_compact[i + sizex * j]) > 0.0001) {
                  printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                      cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact[i+sizex*j], i, j, mat);
                  return false;
                }

        if (fabs(cc.rho_mat_ave[(i + sizex * j) * Nmats + mat] -
                 ccc.rho_mat_ave_compact[i + sizex * j]) > 0.0001) {
                  printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                      cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], ccc.rho_mat_ave_compact[i+sizex*j], i, j, mat);
                  return false;
                }
              }
            }
          }
          printf("All tests passed!\n");
          return true;
        }
