#ifndef SPMV_H
#define SPMV_H

#define REAL float

void mv_csr_serial( const int num_rows, const int *ptr, const int *indices, const REAL *data, const REAL *x, REAL *y);
long mv_csr_parallel(const int repeat, const int bs, const int num_rows, const REAL *x, const int nnz, REAL *matrix, REAL *y);
long mv_dense_parallel(const int repeat, const int bs, const int num_rows, const REAL *x, REAL* matrix, REAL *y);
float check(REAL *A, REAL *B, int n);
void init_vector(REAL *vector, int m);
void init_matrix(REAL *matrix, int num_rows,int nnz);
void init_csr(int *row_indices, REAL *values, int *col_indices, REAL *matrix, int num_rows, int nnz);

#endif
