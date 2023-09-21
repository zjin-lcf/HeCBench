#include <iostream>
#include <cmath>
#include "symbolic.h"
#include "Timer.h"
#include <sycl/sycl.hpp>

#define TMPMEMNUM  10353

#define Nstreams 16

void RL(
    sycl::nd_item<1> &item,
    REAL* __restrict__ s,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos)
{
  const int tid = item.get_local_id(0);
  const int bid = item.get_group(0);
  const int wid = item.get_local_id(0) / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;


  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += item.get_local_range(0);
  }
  item.barrier(sycl::access::fence_space::local_space);

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = item.get_local_id(0) % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //item.barrier(access::fence_space::local_space);
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            auto val_ref = sycl::atomic_ref<REAL,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space> (val_dev[subColElem]);
            val_ref.fetch_add(-tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += item.get_local_range(0)/32;
  }

  item.barrier(sycl::access::fence_space::local_space);
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += item.get_local_range(0);
  }
}

void RL_perturb(
    sycl::nd_item<1> &item,
    REAL* __restrict__ s,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos,
    const float pert)
{
  const int tid = item.get_local_id(0);
  const int bid = item.get_group(0);
  const int wid = item.get_local_id(0) / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += item.get_local_range(0);
  }
  item.barrier(sycl::access::fence_space::local_space);

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = item.get_local_id(0) % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //item.barrier(access::fence_space::local_space);
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            auto val_ref = sycl::atomic_ref<REAL,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space> (val_dev[subColElem]);
            val_ref.fetch_add(-tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += item.get_local_range(0)/32;
  }

  item.barrier(sycl::access::fence_space::local_space);
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += item.get_local_range(0);
  }
}

void RL_onecol_factorizeCurrentCol(
    sycl::nd_item<1> &item,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  const int tid = item.get_local_id(0);

  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
    }
    offset += item.get_local_range(0);
  }
}

void RL_onecol_factorizeCurrentCol_perturb(
    sycl::nd_item<1> &item,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n,
    const float pert)
{
  const int tid = item.get_local_id(0);

  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
    }
    offset += item.get_local_range(0);
  }
}

void RL_onecol_updateSubmat(
    sycl::nd_item<1> &item,
    REAL* __restrict__ s,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  const int tid = item.get_local_id(0);
  const int bid = item.get_group(0);

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
  unsigned subCol;
  unsigned subColElem = 0;

  int offset = 0;
  subCol = csr_c_idx_dev[subColPos];
  while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
  {
    if (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
      subColElem = sym_c_ptr_dev[subCol] + tid + offset;
      unsigned ridx = sym_r_idx_dev[subColElem];

      if (ridx == currentCol)
      {
        s[0] = val_dev[subColElem];
      }
      item.barrier(sycl::access::fence_space::local_space);
      if (ridx > currentCol)
      {
        auto val_ref = sycl::atomic_ref<REAL,
                       sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space> (val_dev[subColElem]);
        val_ref.fetch_add(-tmpMem[stream * n + ridx]*s[0]);
      }
    }
    offset += item.get_local_range(0);
  }
}

void RL_onecol_cleartmpMem(
    sycl::nd_item<1> &item,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  const int tid = item.get_local_id(0);

  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  unsigned offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[stream * n + ridx]= 0;
    }
    offset += item.get_local_range(0);
  }
}

void LUonDevice(Symbolic_Matrix &A_sym, std::ostream &out, std::ostream &err, bool PERTURB)
{
  unsigned n = A_sym.n;
  unsigned nnz = A_sym.nnz;
  unsigned num_lev = A_sym.num_lev;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned *sym_c_ptr_dev = sycl::malloc_device<unsigned>(n + 1, q);
  unsigned *sym_r_idx_dev = sycl::malloc_device<unsigned>(nnz, q );
      REAL *      val_dev = sycl::malloc_device<REAL>(nnz, q);
  unsigned *l_col_ptr_dev = sycl::malloc_device<unsigned>(n, q);
  unsigned *csr_r_ptr_dev = sycl::malloc_device<unsigned>(n + 1, q);
  unsigned *csr_c_idx_dev = sycl::malloc_device<unsigned>(nnz, q);
  unsigned *csr_diag_ptr_dev = sycl::malloc_device<unsigned>(n, q);
       int *level_idx_dev = sycl::malloc_device<int>(n, q);

  q.memcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned));
  q.memcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned));
  q.memcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL));
  q.memcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned));
  q.memcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned));
  q.memcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned));
  q.memcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned));
  q.memcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int));

  REAL *tmpMem = sycl::malloc_device<REAL>(TMPMEMNUM * n, q);
  q.memset(tmpMem, 0, TMPMEMNUM*n*sizeof(REAL));

  // calculate 1-norm of A and perturbation value for perturbation
  float pert = 0;
  if (PERTURB)
  {
    float norm_A = 0;
    for (unsigned i = 0; i < n; ++i)
    {
      float tmp = 0;
      for (unsigned j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
        tmp += abs(A_sym.val[j]);
      if (norm_A < tmp)
        norm_A = tmp;
    }
    pert = 3.45e-4 * norm_A;
    out << "Gaussian elimination with static pivoting (GESP)..." << std::endl;
    out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << std::endl;
  }

  q.wait();

  Timer t;
  double utime;
  t.start();
  for (unsigned i = 0; i < num_lev; ++i)
  {
    // level head
    // note: sycl compile error when A_sym.level_ptr[i] is a kernel argument
    int l = A_sym.level_ptr[i];

    int lev_size = A_sym.level_ptr[i + 1] - l;

    if (lev_size > 896) { //3584 / 4
      unsigned WarpsPerBlock = 2;
      sycl::range<1> lws (WarpsPerBlock * 32);

      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        sycl::range<1> gws (restCol * WarpsPerBlock * 32);
        if (!PERTURB)
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<REAL> sm (sycl::range<1>(WarpsPerBlock), cgh);
            cgh.parallel_for<class RLk>(
              sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              RL(item,
                 sm.get_pointer(),
                 sym_c_ptr_dev,
                 sym_r_idx_dev,
                 val_dev,
                 l_col_ptr_dev,
                 csr_r_ptr_dev,
                 csr_c_idx_dev,
                 csr_diag_ptr_dev,
                 level_idx_dev,
                 tmpMem,
                 n,
                 l,
                 j*TMPMEMNUM);
            });
          });
        else
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<REAL> sm (sycl::range<1>(WarpsPerBlock), cgh);
            cgh.parallel_for<class RL_pk>(
              sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              RL_perturb(
                item,
                sm.get_pointer(),
                sym_c_ptr_dev,
                sym_r_idx_dev,
                val_dev,
                l_col_ptr_dev,
                csr_r_ptr_dev,
                csr_c_idx_dev,
                csr_diag_ptr_dev,
                level_idx_dev,
                tmpMem,
                n,
                l,
                j*TMPMEMNUM,
                pert);
            });
          });
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > 448) {
      unsigned WarpsPerBlock = 4;
      sycl::range<1> lws (WarpsPerBlock * 32);

      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        sycl::range<1> gws (restCol * WarpsPerBlock * 32);
        if (!PERTURB)
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<REAL> sm (sycl::range<1>(WarpsPerBlock), cgh);
            cgh.parallel_for<class RLk2>(
              sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              RL(item,
                 sm.get_pointer(),
                 sym_c_ptr_dev,
                 sym_r_idx_dev,
                 val_dev,
                 l_col_ptr_dev,
                 csr_r_ptr_dev,
                 csr_c_idx_dev,
                 csr_diag_ptr_dev,
                 level_idx_dev,
                 tmpMem,
                 n,
                 l,
                 j*TMPMEMNUM);
            });
          });
        else
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<REAL> sm (sycl::range<1>(WarpsPerBlock), cgh);
            cgh.parallel_for<class RL_pk2>(
              sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              RL_perturb(
                item,
                sm.get_pointer(),
                sym_c_ptr_dev,
                sym_r_idx_dev,
                val_dev,
                l_col_ptr_dev,
                csr_r_ptr_dev,
                csr_c_idx_dev,
                csr_diag_ptr_dev,
                level_idx_dev,
                tmpMem,
                n,
                l,
                j*TMPMEMNUM,
                pert);
            });
          });
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > Nstreams) {
      unsigned WarpsPerBlock = 32;
      sycl::range<1> lws (256);
      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        sycl::range<1> gws (restCol * 256);
        if (!PERTURB)
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<REAL> sm (sycl::range<1>(WarpsPerBlock), cgh);
            cgh.parallel_for<class RLk3>(
              sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              RL(item,
                 sm.get_pointer(),
                 sym_c_ptr_dev,
                 sym_r_idx_dev,
                 val_dev,
                 l_col_ptr_dev,
                 csr_r_ptr_dev,
                 csr_c_idx_dev,
                 csr_diag_ptr_dev,
                 level_idx_dev,
                 tmpMem,
                 n,
                 l,
                 j*TMPMEMNUM);
            });
          });
        else
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<REAL> sm (sycl::range<1>(WarpsPerBlock), cgh);
            cgh.parallel_for<class RL_pk3>(
              sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              RL_perturb(
                item,
                sm.get_pointer(),
                sym_c_ptr_dev,
                sym_r_idx_dev,
                val_dev,
                l_col_ptr_dev,
                csr_r_ptr_dev,
                csr_c_idx_dev,
                csr_diag_ptr_dev,
                level_idx_dev,
                tmpMem,
                n,
                l,
                j*TMPMEMNUM,
                pert);
             });
           });
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else { // "Big" levels
      for (int offset = 0; offset < lev_size; offset += Nstreams) {
        for (int j = 0; j < Nstreams; j++) {
          if (j + offset < lev_size) {
            const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
            const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
              - A_sym.csr_diag_ptr[currentCol] - 1;

            if (!PERTURB)
              q.submit([&] (sycl::handler &cgh) {
                cgh.parallel_for<class factorizeCol>(
                  sycl::nd_range<1>(256, 256), [=] (sycl::nd_item<1> item) {
                  RL_onecol_factorizeCurrentCol(
                    item,
                    sym_c_ptr_dev,
                    sym_r_idx_dev,
                    val_dev,
                    l_col_ptr_dev,
                    currentCol,
                    tmpMem,
                    j,
                    n);
                });
              });
            else
              q.submit([&] (sycl::handler &cgh) {
                cgh.parallel_for<class factorizeColPerturb>(
                  sycl::nd_range<1>(256, 256), [=] (sycl::nd_item<1> item) {
                  RL_onecol_factorizeCurrentCol_perturb(
                    item,
                    sym_c_ptr_dev,
                    sym_r_idx_dev,
                    val_dev,
                    l_col_ptr_dev,
                    currentCol,
                    tmpMem,
                    j,
                    n,
                    pert);
                });
              });

            if (subMatSize > 0)
              q.submit([&] (sycl::handler &cgh) {
                sycl::local_accessor<REAL> sm (1, cgh);
                cgh.parallel_for<class update>(
                  sycl::nd_range<1>(256*subMatSize, 256), [=] (sycl::nd_item<1> item) {
                  RL_onecol_updateSubmat(
                    item,
                    sm.get_pointer(),
                    sym_c_ptr_dev,
                    sym_r_idx_dev,
                    val_dev,
                    csr_c_idx_dev,
                    csr_diag_ptr_dev,
                    currentCol,
                    tmpMem,
                    j,
                    n);
                });
              });

            q.submit([&] (sycl::handler &cgh) {
              cgh.parallel_for<class clearMem>(
                sycl::nd_range<1>(256, 256), [=] (sycl::nd_item<1> item) {
                RL_onecol_cleartmpMem(
                  item,
                  sym_c_ptr_dev,
                  sym_r_idx_dev,
                  l_col_ptr_dev,
                  currentCol,
                  tmpMem,
                  j,
                  n);
              });
            });
          }
        }
      }
    }
  }

  q.wait();
  t.elapsedUserTime(utime);
  out << "Total LU kernel execution time: " << utime << " ms" << std::endl;

  //copy LU val back to main mem
  q.memcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL)).wait();

#ifdef VERIFY
  //check NaN elements
  unsigned err_find = 0;
  for(unsigned i = 0; i < nnz; i++)
    if(std::isnan(A_sym.val[i]) || std::isinf(A_sym.val[i]))
      err_find++;

  if (err_find != 0)
    err << "LU data check: NaN found!!" << std::endl;
#endif

  sycl::free(sym_c_ptr_dev, q);
  sycl::free(sym_r_idx_dev, q);
  sycl::free(val_dev, q);
  sycl::free(l_col_ptr_dev, q);
  sycl::free(csr_c_idx_dev, q);
  sycl::free(csr_r_ptr_dev, q);
  sycl::free(csr_diag_ptr_dev, q);
  sycl::free(level_idx_dev, q);
  sycl::free(tmpMem, q);
}
