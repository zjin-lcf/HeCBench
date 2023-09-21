#include <chrono>
#include <math.h>
#include <stdio.h>
#include <sycl/sycl.hpp>

struct full_data
{
  int sizex;
  int sizey;
  int Nmats;
  double * __restrict rho;
  double * __restrict rho_mat_ave;
  double * __restrict p;
  double * __restrict Vf;
  double * __restrict t;
  double * __restrict V;
  double * __restrict x;
  double * __restrict y;
  double * __restrict n;
  double * __restrict rho_ave;
};

struct compact_data
{
  int sizex;
  int sizey;
  int Nmats;
  double * __restrict rho_compact;
  double * __restrict rho_compact_list;
  double * __restrict rho_mat_ave_compact;
  double * __restrict rho_mat_ave_compact_list;
  double * __restrict p_compact;
  double * __restrict p_compact_list;
  double * __restrict Vf_compact_list;
  double * __restrict t_compact;
  double * __restrict t_compact_list;
  double * __restrict V;
  double * __restrict x;
  double * __restrict y;
  double * __restrict n;
  double * __restrict rho_ave_compact;
  int * __restrict imaterial;
  int * __restrict matids;
  int * __restrict nextfrac;
  int * __restrict mmc_index;
  int * __restrict mmc_i;
  int * __restrict mmc_j;
  int mm_len;
  int mmc_cells;
};

char *cp_to_device(sycl::queue &q, char *from, size_t size) {
  char *tmp = (char*) sycl::malloc_device(size, q);
  q.memcpy(tmp, from, size);
  return tmp;
}

void cp_to_host(sycl::queue &q, char *to, char *from, size_t size) {
  q.memcpy(to, from, size).wait();
  sycl::free(from, q);
}

void compact_cell_centric(full_data cc, compact_data ccc, int argc, char** argv)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int sizex = cc.sizex;
  int sizey = cc.sizey;
  int Nmats = cc.Nmats;
  int mmc_cells = ccc.mmc_cells;
  int mm_len = ccc.mm_len;

  int    *d_imaterial = (int *)cp_to_device(q, (char*)ccc.imaterial, sizex*sizey*sizeof(int));

  int    *d_matids = (int *)cp_to_device(q, (char*)ccc.matids, mm_len*sizeof(int));

  int    *d_nextfrac = (int *)cp_to_device(q, (char*)ccc.nextfrac, mm_len*sizeof(int));

  int    *d_mmc_index = (int *)cp_to_device(q, (char*)ccc.mmc_index, (mmc_cells+1)*sizeof(int));

  int    *d_mmc_i = (int *)cp_to_device(q, (char*)ccc.mmc_i, (mmc_cells)*sizeof(int));

  int    *d_mmc_j = (int *)cp_to_device(q, (char*)ccc.mmc_j, (mmc_cells)*sizeof(int));

  double *d_x = (double *)cp_to_device(q, (char*)ccc.x, sizex*sizey*sizeof(double));

  double *d_y = (double *)cp_to_device(q, (char*)ccc.y, sizex*sizey*sizeof(double));

  double *d_rho_compact = (double *)cp_to_device(q, (char*)ccc.rho_compact, sizex*sizey*sizeof(double));

  double *d_rho_compact_list = (double *)cp_to_device(q, (char*)ccc.rho_compact_list,mm_len*sizeof(double));

  double *d_rho_mat_ave_compact = (double *)cp_to_device(q, (char*)ccc.rho_mat_ave_compact, sizex*sizey*sizeof(double));

  double *d_rho_mat_ave_compact_list = (double *)cp_to_device(q, (char*)ccc.rho_mat_ave_compact_list,mm_len*sizeof(double));

  double *d_p_compact = (double *)cp_to_device(q, (char*)ccc.p_compact, sizex*sizey*sizeof(double));

  double *d_p_compact_list = (double *)cp_to_device(q, (char*)ccc.p_compact_list,mm_len*sizeof(double));

  double *d_t_compact = (double *)cp_to_device(q, (char*)ccc.t_compact, sizex*sizey*sizeof(double));

  double *d_t_compact_list = (double *)cp_to_device(q, (char*)ccc.t_compact_list,mm_len*sizeof(double));

  double *d_V = (double *)cp_to_device(q, (char*)ccc.V, sizex*sizey*sizeof(double));

  double *d_Vf_compact_list = (double *)cp_to_device(q, (char*)ccc.Vf_compact_list, mm_len*sizeof(double));

  double *d_n = (double *)cp_to_device(q, (char*)ccc.n, Nmats*sizeof(double));

  double *d_rho_ave_compact = (double *)cp_to_device(q, (char*)ccc.rho_ave_compact, sizex*sizey*sizeof(double));

  const int thx = 32;
  const int thy = 4;

  sycl::range<2> ccc_loop1_gws ((sizey+thy-1)/thy*thy, (sizex+thx-1)/thx*thx);
  sycl::range<2> ccc_loop1_lws (thy, thx);

  // Cell-centric algorithms
  // Computational loop 1 - average density in cell
  q.wait();

  auto t0 = std::chrono::system_clock::now();
  //ccc_loop1 <<< dim3(blocks), dim3(threads) >>> (d_imaterial, d_nextfrac, d_rho_compact, d_rho_compact_list, d_Vf_compact_list, d_V, d_rho_ave_compact, sizex, sizey, d_mmc_index);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class ccc_loop1>(
      sycl::nd_range<2>(ccc_loop1_gws, ccc_loop1_lws), [=] (sycl::nd_item<2> item) {
      int i = item.get_global_id(1); 
      int j = item.get_global_id(0);
      if (i >= sizex || j >= sizey) return;
    #ifdef FUSED
      double ave = 0.0;
      int ix = d_imaterial[i+sizex*j];
    
      if (ix <= 0) {
        // condition is 'ix >= 0', this is the equivalent of
        // 'until ix < 0' from the paper
    #ifdef LINKED
        for (ix = -ix; ix >= 0; ix = d_nextfrac[ix]) {
          ave += d_rho_compact_list[ix] * d_Vf_compact_list[ix];
        }
    #else
        for (int idx = d_mmc_index[-ix]; idx < d_mmc_index[-ix+1]; idx++) {
          ave += d_rho_compact_list[idx] * d_Vf_compact_list[idx];  
        }
    #endif
        d_rho_ave_compact[i+sizex*j] = ave / d_V[i+sizex*j];
      }
      else {
    #endif
        // We use a distinct output array for averages.
        // In case of a pure cell, the average density equals to the total.
        d_rho_ave_compact[i+sizex*j] = d_rho_compact[i+sizex*j] / d_V[i+sizex*j];
    #ifdef FUSED
      }
    #endif
     
    });
    });

#ifndef FUSED

  sycl::range<1> ccc_loop1_2_gws ((mmc_cells + thx * thy - 1)/(thx * thy) * (thx*thy));
  sycl::range<1> ccc_loop1_2_lws (thx*thy);

  // ccc_loop1_2 <<< dim3((mmc_cells-1)/(thx*thy)+1), dim3((thx*thy)) >>> (d_rho_compact_list, d_Vf_compact_list, d_V, d_rho_ave_compact, d_mmc_index, mmc_cells, d_mmc_i, d_mmc_j, sizex, sizey);
    q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class ccc_loop1_2>(
      sycl::nd_range<1>(ccc_loop1_2_gws, ccc_loop1_2_lws), [=] (sycl::nd_item<1> item) {
      int c = item.get_global_id(0);
      if (c >= mmc_cells) return;
      double ave = 0.0;
      for (int m = d_mmc_index[c]; m < d_mmc_index[c+1]; m++) {
        ave += d_rho_compact_list[m] * d_Vf_compact_list[m];
      }
      d_rho_ave_compact[d_mmc_i[c] + sizex * d_mmc_j[c]] = ave / d_V[d_mmc_i[c] + sizex * d_mmc_j[c]];
    });
  });

#endif
  q.wait();
  std::chrono::duration<double> t1 = std::chrono::system_clock::now() - t0;
  printf("Compact matrix, cell centric, alg 1: %g msec\n", t1.count() * 1000);

  // Computational loop 2 - Pressure for each cell and each material
  t0 = std::chrono::system_clock::now();
  // ccc_loop2 <<< dim3(blocks), dim3(threads) >>> (d_imaterial, d_matids,d_nextfrac, d_rho_compact, d_rho_compact_list, d_t_compact, d_t_compact_list, d_Vf_compact_list, d_n, d_p_compact, d_p_compact_list, sizex, sizey, d_mmc_index);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class ccc_loop2>(
      sycl::nd_range<2>(ccc_loop1_gws, ccc_loop1_lws), [=] (sycl::nd_item<2> item) {
  
      int i = item.get_global_id(1); 
      int j = item.get_global_id(0);
      if (i >= sizex || j >= sizey) return;
      int ix = d_imaterial[i+sizex*j];
      if (ix <= 0) {
#ifdef FUSED
    // NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
    // The solution below is what I believe to good.

    // condition is 'ix >= 0', this is the equivalent of
    // 'until ix < 0' from the paper
#ifdef LINKED
      for (ix = -ix; ix >= 0; ix = d_nextfrac[ix]) {
        double nm = d_n[d_matids[ix]];
        d_p_compact_list[ix] = (nm * d_rho_compact_list[ix] * d_t_compact_list[ix]) / d_Vf_compact_list[ix];
      }
#else
      for (int idx = d_mmc_index[-ix]; idx < d_mmc_index[-ix+1]; idx++) {
        double nm = d_n[d_matids[idx]];
        d_p_compact_list[idx] = (nm * d_rho_compact_list[idx] * d_t_compact_list[idx]) / d_Vf_compact_list[idx];
      }
#endif
#endif
     }
     else {
       // NOTE: HACK: we index materials from zero, but zero can be a list index
       int mat = ix - 1;
       // NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
       d_p_compact[i+sizex*j] = d_n[mat] * d_rho_compact[i+sizex*j] * d_t_compact[i+sizex*j];;
     }
    });
  });

#ifndef FUSED
  sycl::range<1> ccc_loop2_2_gws ((mm_len + thx * thy - 1)/(thx * thy) * (thx*thy));
  sycl::range<1> ccc_loop2_2_lws (thx*thy);
  //ccc_loop2_2 <<< dim3((mm_len-1)/(thx*thy)+1), dim3((thx*thy)) >>> (d_matids, d_rho_compact_list, d_t_compact_list, d_Vf_compact_list, d_n, d_p_compact_list, d_mmc_index, mm_len);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class ccc_loop2_2>(
      sycl::nd_range<1>(ccc_loop2_2_gws, ccc_loop2_2_lws), [=] (sycl::nd_item<1> item) {
      int idx = item.get_global_id(0);
      if (idx >= mm_len) return;
      double nm = d_n[d_matids[idx]];
      d_p_compact_list[idx] = (nm * d_rho_compact_list[idx] * d_t_compact_list[idx]) / d_Vf_compact_list[idx];
    });
  });
  #endif

  q.wait();
  std::chrono::duration<double> t2 = std::chrono::system_clock::now() - t0;
  printf("Compact matrix, cell centric, alg 2: %g msec\n", t2.count() * 1000);

  // Computational loop 3 - Average density of each material over neighborhood of each cell
  t0 = std::chrono::system_clock::now();
  //ccc_loop3 <<< dim3(blocks), dim3(threads) >>> (d_imaterial,d_nextfrac, d_matids, d_rho_compact, d_rho_compact_list, d_rho_mat_ave_compact, d_rho_mat_ave_compact_list, d_x, d_y, sizex, sizey, d_mmc_index);  
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class ccc_loop3>(
      sycl::nd_range<2>(ccc_loop1_gws, ccc_loop1_lws), [=] (sycl::nd_item<2> item) {
      int i = item.get_global_id(1); 
      int j = item.get_global_id(0);
      if (i >= sizex-1 || j >= sizey-1 || i < 1 || j < 1) return;

      double xo = d_x[i+sizex*j];
      double yo = d_y[i+sizex*j];

      // There are at most 9 neighbours in 2D case.
      double dsqr[9];

      // for all neighbours
      for (int nj = -1; nj <= 1; nj++) {

        for (int ni = -1; ni <= 1; ni++) {

          dsqr[(nj+1)*3 + (ni+1)] = 0.0;

          // i: inner
          double xi = d_x[(i+ni)+sizex*(j+nj)];
          double yi = d_y[(i+ni)+sizex*(j+nj)];

          dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
          dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
        }
      }

      int ix = d_imaterial[i+sizex*j];

      if (ix <= 0) {

#ifdef LINKED
        for (ix = -ix; ix >= 0; ix = d_nextfrac[ix]) {
#else
        for (int ix = d_mmc_index[-d_imaterial[i+sizex*j]]; ix < d_mmc_index[-d_imaterial[i+sizex*j]+1]; ix++) {
#endif
          int mat = d_matids[ix];
          double rho_sum = 0.0;
          int Nn = 0;

          // for all neighbours
          for (int nj = -1; nj <= 1; nj++) {
            for (int ni = -1; ni <= 1; ni++) {
              int ci = i+ni, cj = j+nj;
              int jx = d_imaterial[ci+sizex*cj];

              if (jx <= 0) {
#ifdef LINKED
                for (jx = -jx; jx >= 0; jx = d_nextfrac[jx]) {
#else
                for (int jx = d_mmc_index[-d_imaterial[ci+sizex*cj]]; jx < d_mmc_index[-d_imaterial[ci+sizex*cj]+1]; jx++) {
#endif
                  if (d_matids[jx] == mat) {
                    rho_sum += d_rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
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
                  rho_sum += d_rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
                  Nn += 1;
                }
              } // end if (jx <= 0)
        } // end for (int ni)
      } // end for (int nj)

      d_rho_mat_ave_compact_list[ix] = rho_sum / Nn;
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
         int jx = d_imaterial[ci+sizex*cj];

         if (jx <= 0) {
           // condition is 'jx >= 0', this is the equivalent of
           // 'until jx < 0' from the paper
#ifdef LINKED
            for (jx = -jx; jx >= 0; jx = d_nextfrac[jx]) {
#else
            for (int jx = d_mmc_index[-d_imaterial[ci+sizex*cj]]; jx < d_mmc_index[-d_imaterial[ci+sizex*cj]+1]; jx++) {
#endif
            if (d_matids[jx] == mat) {
              rho_sum += d_rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
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
            rho_sum += d_rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
            Nn += 1;
          }
        } // end if (jx <= 0)
      } // end for (int ni)
    } // end for (int nj)

    d_rho_mat_ave_compact[i+sizex*j] = rho_sum / Nn;
   } // end else
  });
  });
  q.wait();
  std::chrono::duration<double> t3 = std::chrono::system_clock::now() - t0;
  printf("Compact matrix, cell centric, alg 3: %g msec\n", t3.count() * 1000);

  cp_to_host(q, (char*)ccc.x, (char*)d_x, sizex*sizey*sizeof(double));
  cp_to_host(q, (char*)ccc.y, (char*)d_y, sizex*sizey*sizeof(double));
  cp_to_host(q, (char*)ccc.rho_compact, (char*)d_rho_compact, sizex*sizey*sizeof(double));
  cp_to_host(q, (char*)ccc.rho_compact_list, (char*)d_rho_compact_list, mm_len*sizeof(double));
  cp_to_host(q, (char*)ccc.rho_mat_ave_compact, (char*)d_rho_mat_ave_compact, sizex*sizey*sizeof(double));
  cp_to_host(q, (char*)ccc.rho_mat_ave_compact_list, (char*)d_rho_mat_ave_compact_list, mm_len*sizeof(double));
  cp_to_host(q, (char*)ccc.p_compact, (char*)d_p_compact, sizex*sizey*sizeof(double));
  cp_to_host(q, (char*)ccc.p_compact_list, (char*)d_p_compact_list, mm_len*sizeof(double));
  cp_to_host(q, (char*)ccc.t_compact, (char*)d_t_compact, sizex*sizey*sizeof(double));
  cp_to_host(q, (char*)ccc.t_compact_list, (char*)d_t_compact_list, mm_len*sizeof(double));
  cp_to_host(q, (char*)ccc.Vf_compact_list, (char*)d_Vf_compact_list, mm_len*sizeof(double));
  cp_to_host(q, (char*)ccc.V, (char*)d_V, sizex*sizey*sizeof(double));
  cp_to_host(q, (char*)ccc.n, (char*)d_n, Nmats*sizeof(double));
  cp_to_host(q, (char*)ccc.rho_ave_compact, (char*)d_rho_ave_compact, sizex*sizey*sizeof(double));
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
      if (fabs(cc.rho_ave[i+sizex*j] - ccc.rho_ave_compact[i+sizex*j]) > 0.0001) {
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
            if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - ccc.p_compact_list[ix]) > 0.0001) {
              printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                  cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact_list[ix], i, j, mat);
              return false;
            }

            if (fabs(cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] - ccc.rho_mat_ave_compact_list[ix]) > 0.0001) {
              printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                  cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], ccc.rho_mat_ave_compact_list[ix], i, j, mat);
              return false;
            }
          }
        }
        else {
          // NOTE: HACK: we index materials from zero, but zero can be a list index
          int mat = ix - 1;
          if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - ccc.p_compact[i+sizex*j]) > 0.0001) {
            printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact[i+sizex*j], i, j, mat);
            return false;
          }

          if (fabs(cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] - ccc.rho_mat_ave_compact[i+sizex*j]) > 0.0001) {
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
