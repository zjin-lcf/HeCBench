#include <iostream>
#include <cmath>
#include <omp.h>
#include "symbolic.h"
#include "Timer.h"

using namespace std;

#define TMPMEMNUM  10353
#define Nstreams   16

void RL(
    const int nteams,
    const int nthreads,
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
  #pragma omp target teams num_teams(nteams) thread_limit(nthreads)
  {
    REAL s[32];
    #pragma omp parallel 
    {
      const int tid = omp_get_thread_num();
      const int bid = omp_get_team_num();
      const int wid = omp_get_thread_num() / 32;

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
        offset += omp_get_num_threads();
      }
      #pragma omp barrier

      //broadcast to submatrix
      const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
      const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
      unsigned subCol;
      const int tidInWarp = omp_get_thread_num() % 32;
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
              //#pragma omp barrier
              if (ridx > currentCol)
              {
                //elem in currentCol same row with subColElem might be 0, so
                //clearing tmpMem is necessary
                #pragma omp atomic update
                val_dev[subColElem] += -tmpMem[ridx+n*bid]*s[wid];
              }
            }
            offset += 32;
          }
        }
        woffset += omp_get_num_threads()/32;
      }

      #pragma omp barrier
      //Clear tmpMem
      offset = 0;
      while (currentLColSize > offset)
      {
        if (tid + offset < currentLColSize)
        {
          unsigned ridx = sym_r_idx_dev[currentLPos + offset];
          tmpMem[bid*n + ridx]= 0;
        }
        offset += omp_get_num_threads();
      }
    }
  }
}

void RL_perturb(
    const int nteams,
    const int nthreads,
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
  #pragma omp target teams num_teams(nteams) thread_limit(nthreads)
  {
    REAL s[32];
    #pragma omp parallel 
    {
      const int tid = omp_get_thread_num();
      const int bid = omp_get_team_num();
      const int wid = omp_get_thread_num() / 32;

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
        offset += omp_get_num_threads();
      }
      #pragma omp barrier

      //broadcast to submatrix
      const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
      const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
      unsigned subCol;
      const int tidInWarp = omp_get_thread_num() % 32;
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
              //#pragma omp barrier
              if (ridx > currentCol)
              {
                //elem in currentCol same row with subColElem might be 0, so
                //clearing tmpMem is necessary
                #pragma omp atomic update
                val_dev[subColElem] += -tmpMem[ridx+n*bid]*s[wid];
              }
            }
            offset += 32;
          }
        }
        woffset += omp_get_num_threads()/32;
      }

      #pragma omp barrier
      //Clear tmpMem
      offset = 0;
      while (currentLColSize > offset)
      {
        if (tid + offset < currentLColSize)
        {
          unsigned ridx = sym_r_idx_dev[currentLPos + offset];
          tmpMem[bid*n + ridx]= 0;
        }
        offset += omp_get_num_threads();
      }
    }
  }
}

void RL_onecol_factorizeCurrentCol(
    const int nteams,
    const int nthreads,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  #pragma omp target teams num_teams(nteams) thread_limit(nthreads)
  {
    #pragma omp parallel 
    {
      const int tid = omp_get_thread_num();

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
        offset += omp_get_num_threads();
      }
    }
  }
}

void RL_onecol_factorizeCurrentCol_perturb(
    const int nteams,
    const int nthreads,
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
  #pragma omp target teams num_teams(nteams) thread_limit(nthreads)
  {
    #pragma omp parallel 
    {
      const int tid = omp_get_thread_num();

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
        offset += omp_get_num_threads();
      }
    }
  }
}

void RL_onecol_updateSubmat(
    const int nteams,
    const int nthreads,
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
  #pragma omp target teams num_teams(nteams) thread_limit(nthreads)
  {
    REAL s;
    #pragma omp parallel 
    {
      const int tid = omp_get_thread_num();
      const int bid = omp_get_team_num();

      //broadcast to submatrix
      const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
      unsigned subColElem = 0;
      unsigned ridx;

      int offset = 0;
      unsigned subCol = csr_c_idx_dev[subColPos];
      const int range = sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol];
      while(offset < range)
      {
        if (tid + offset < range)
        {
          subColElem = sym_c_ptr_dev[subCol] + tid + offset;
          ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s = val_dev[subColElem];
          }
        }
        #pragma omp barrier

        if (tid + offset < range)
        {
          if (ridx > currentCol)
          {
            #pragma omp atomic update
            val_dev[subColElem] += -tmpMem[stream * n + ridx] * s;
          }
        }
        offset += omp_get_num_threads();
      }
    }
  }
}

void RL_onecol_cleartmpMem(
    const int nteams,
    const int nthreads,
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  #pragma omp target teams num_teams(nteams) thread_limit(nthreads)
  {
    #pragma omp parallel 
    {
      const int tid = omp_get_thread_num();

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
        offset += omp_get_num_threads();
      }
    }
  }
}

void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB)
{
  unsigned n = A_sym.n;
  unsigned nnz = A_sym.nnz;
  unsigned num_lev = A_sym.num_lev;

  unsigned *sym_c_ptr_dev = &(A_sym.sym_c_ptr[0]);
  unsigned *sym_r_idx_dev = &(A_sym.sym_r_idx[0]);
  REAL *val_dev = &(A_sym.val[0]);
  unsigned *l_col_ptr_dev = &(A_sym.l_col_ptr[0]);
  unsigned *csr_r_ptr_dev = &(A_sym.csr_r_ptr[0]);
  unsigned *csr_c_idx_dev = &(A_sym.csr_c_idx[0]);
  unsigned *csr_diag_ptr_dev = &(A_sym.csr_diag_ptr[0]);
  int *level_idx_dev = &(A_sym.level_idx[0]);
  REAL *tmpMem = (REAL*) malloc (TMPMEMNUM*n*sizeof(REAL));

  #pragma omp target data map(to: sym_c_ptr_dev[0:n+1],\
                                  sym_r_idx_dev[0:nnz],\
                                  val_dev[0:nnz],\
                                  l_col_ptr_dev[0:n],\
                                  csr_r_ptr_dev[0:n+1],\
                                  csr_c_idx_dev[0:nnz],\
                                  csr_diag_ptr_dev[0:n],\
                                  level_idx_dev[0:n]) \
                          map(alloc: tmpMem[0:TMPMEMNUM*n])
  {

  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < TMPMEMNUM*n; i++)
    tmpMem[i] = (REAL)0;

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
    out << "Gaussian elimination with static pivoting (GESP)..." << endl;
    out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
  }

  Timer t;
  double utime;
  t.start();
  for (unsigned i = 0; i < num_lev; ++i)
  {
    int l = A_sym.level_ptr[i];
    int lev_size = A_sym.level_ptr[i + 1] - l;

    if (lev_size > 896) { //3584 / 4
      int dimBlock = 64;
      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        int dimGrid = restCol;
        if (!PERTURB)
          RL(
              dimGrid, dimBlock,
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
        else
          RL_perturb(
              dimGrid, dimBlock,
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
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > 448) {
      int dimBlock = 128;
      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        int dimGrid = restCol;
        if (!PERTURB)
          RL(
              dimGrid, dimBlock,
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
        else
          RL_perturb(
              dimGrid, dimBlock,
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
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > Nstreams) {
      int dimBlock = 256;
      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        int dimGrid = restCol;
        if (!PERTURB)
          RL(
              dimGrid, dimBlock,
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
        else
          RL_perturb(
              dimGrid, dimBlock,
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
              RL_onecol_factorizeCurrentCol(
                  1, 256,
                  sym_c_ptr_dev,
                  sym_r_idx_dev,
                  val_dev,
                  l_col_ptr_dev,
                  currentCol,
                  tmpMem,
                  j,
                  n);
            else
              RL_onecol_factorizeCurrentCol_perturb(
                  1, 256,
                  sym_c_ptr_dev,
                  sym_r_idx_dev,
                  val_dev,
                  l_col_ptr_dev,
                  currentCol,
                  tmpMem,
                  j,
                  n,
                  pert);
            if (subMatSize > 0)
              RL_onecol_updateSubmat(
                  subMatSize, 256,
                  sym_c_ptr_dev,
                  sym_r_idx_dev,
                  val_dev,
                  csr_c_idx_dev,
                  csr_diag_ptr_dev,
                  currentCol,
                  tmpMem,
                  j,
                  n);
            RL_onecol_cleartmpMem(
                1, 256,
                sym_c_ptr_dev,
                sym_r_idx_dev,
                l_col_ptr_dev,
                currentCol,
                tmpMem,
                j,
                n);
          }
        }
      }
    }
  }
  t.elapsedUserTime(utime);
  out << "Total LU kernel execution time: " << utime << " ms" << std::endl;

  #pragma omp target update from (val_dev[0:nnz])

#ifdef VERIFY
  //check NaN elements
  unsigned err_find = 0;
  for(unsigned i = 0; i < nnz; i++)
    if(isnan(A_sym.val[i]) || isinf(A_sym.val[i])) 
      err_find++;

  if (err_find != 0)
    err << "LU data check: NaN found!!" << std::endl;
#endif

  } // omp
  free(tmpMem);
}
