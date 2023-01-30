#include <stdio.h>
#include <stdlib.h>
#include "mv.h"

int main(int argc, char *argv[])
{
  int nnz = 0;
  int num_rows = 0;
  int repeat = 0;

  if (argc != 4) {
    printf("Usage %s <number of non-zero elements> <number of rows in a square matrix> <repeat>\n", argv[0]);
    printf("The number of non-zero elements and the number of rows in a square matrix must be positive\n");
    return 1;
  }

  nnz = atoi(argv[1]);
  num_rows = atoi(argv[2]);
  repeat = atoi(argv[3]);

  if (nnz <= 0 || num_rows <= 0) return 1;

  int *row_col_indices, *col_indices;
  REAL *values, *x, *y, *y_csr, *y_dense, *y_warmup;
  REAL *matrix;
  row_col_indices = (int *) malloc((num_rows+1) * sizeof(int));
  col_indices = (int *) malloc(nnz * sizeof(int));
  values = (REAL *) malloc(nnz * sizeof(REAL));
  x = (REAL *) malloc(num_rows * sizeof(REAL));
  y = (REAL *) malloc(num_rows * sizeof(REAL));
  y_csr = (REAL *) malloc(num_rows * sizeof(REAL));
  y_dense = (REAL *) malloc(num_rows * sizeof(REAL));
  y_warmup = (REAL *) malloc(num_rows * sizeof(REAL));
  matrix = (REAL *) malloc(num_rows * num_rows * sizeof(REAL));

  srand48(1<<12);
  init_matrix(matrix, num_rows, nnz);
  init_vector(x, num_rows);
  init_csr(row_col_indices, values, col_indices, matrix, num_rows, nnz);

  // reference results in y
  mv_csr_serial(num_rows, row_col_indices, col_indices, values, x, y);

  printf("Number of rows in a square matrix: %d\n", num_rows);
  printf("Number of non-zero elements: %d\n", nnz);

  // thread block size
  for (int bs = 32; bs <= 1024; bs = bs * 2) {
    printf("\nThread block size: %d\n", bs);

    // warmup run
    mv_dense_parallel(1, bs, num_rows, x, matrix, y_warmup);

    long elapsed = mv_dense_parallel(repeat, bs, num_rows, x, matrix, y_dense);

    printf("Average dense kernel execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
    printf("Error rate: %f\n", check(y, y_dense, num_rows));
  }

  for (int bs = 32; bs <= 1024; bs = bs * 2) {
    printf("\nThread block size: %d\n", bs);
    // warmup run
    mv_csr_parallel(1, bs, num_rows, x, nnz, matrix, y_warmup);

    long elapsed = mv_csr_parallel(repeat, bs, num_rows, x, nnz, matrix, y_csr);

    printf("Average csr kernel execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
    printf("Error rate: %f\n", check(y, y_csr, num_rows));
  }

  free(row_col_indices);
  free(col_indices);
  free(values);
  free(x);
  free(y);
  free(y_csr);
  free(y_dense);
  free(y_warmup);
  free(matrix);
  return 0;
}
