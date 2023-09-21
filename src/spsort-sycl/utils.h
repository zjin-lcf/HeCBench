/*******************************************************************************
* Copyright 2019-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include <complex>

template <typename fp, typename intType>
void init_matrix(fp *matrix, intType num_rows, intType num_cols, intType nnz)
{
  intType n = (intType)num_rows * num_cols;

  fp *d = (fp *) malloc(n * sizeof(fp));

  srand(123);
  for (intType i = 0; i < n; i++) d[i] = (fp)i;
  for (intType i = n; i > 0; i--) {
    intType a = i-1;
    intType b = rand() % i;
    if (a != b) {
      auto t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  srand48(123);
  for (intType i = 0; i < num_rows; i++) {
    for (intType j = 0; j < num_cols; j++) {
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0 : (fp)(drand48()+1);
    }
  }

  free(d);
}

template <typename fp, typename intType>
void init_csr(intType *row_indices, fp *values,
              intType *col_indices, fp *matrix,
              intType num_rows, intType num_cols, intType nnz)
{
  row_indices[num_rows] = nnz;
  row_indices[0] = 0;
  intType *non_zero_elements = (intType*) malloc (num_rows * sizeof(intType));

  intType tmp = 0;
  for (intType i = 0; i < num_rows; i++) {
    intType nnz_per_row = 0; // nnz per row
    for (intType j = 0; j < num_cols; j++) {
      if(matrix[i*num_cols+j] != 0) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    non_zero_elements[i] = nnz_per_row;
  }

  for (intType i = 1; i < num_rows; i++) {
    row_indices[i] = row_indices[i-1] + non_zero_elements[i-1];
  }

  free(non_zero_elements);
}

// Shuffle the 3arrays CSR representation (ia, ja, values)
// of any sparse matrix.
template <typename fp, typename intType>
void shuffle_matrix_data(const intType *ia,
                         intType *ja,
                         fp *a,
                         const intType nrows,
                         const intType nnz)
{
    //
    // shuffle indices according to random seed
    //
    for (intType i = 0; i < nrows; ++i) {
        intType nnz_row = ia[i+1]-ia[i];
        for (intType j = ia[i]; j < ia[i+1]; ++j) {
            intType q = ia[i] + std::rand() % (nnz_row);
            // swap element i and q
            std::swap(ja[q], ja[j]);
            std::swap(a[q], a[j]);
        }
    }
}

