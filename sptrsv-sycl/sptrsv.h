#ifndef _SPTRSV_H_
#define _SPTRSV_H_

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK 8
#endif

#define SUBSTITUTION_FORWARD  0
#define SUBSTITUTION_BACKWARD 1

#define OPT_WARP_NNZ   1
#define OPT_WARP_RHS   2
#define OPT_WARP_AUTO  3

int sptrsv_syncfree (
    const int           repeat,
    const int           *csrRowPtr,
    const int           *csrColIdx,
    const VALUE_TYPE    *csrVal,
    const int           m,
    const int           n,
    const int           nnz,
    VALUE_TYPE          *x,
    const VALUE_TYPE    *b,
    const VALUE_TYPE    *x_ref);

void matrix_warp4    (const int         m,
                      const int         n,
                      const int         nnz,
                      const int        *csrRowPtr,
                      const int        *csrColIdx,
                      const VALUE_TYPE *csrVal,
                      const int         border,
                      int              *Len_add,
                      int              *warp_num,
                      double           *warp_occupy_add,
                      double           *element_occupy_add
                      );
#endif

