#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
SYCL_EXTERNAL void lud_diagonal(float *m, const int matrix_dim,
                                const int offset, sycl::nd_item<3> item_ct1,
                                float *shadow) {

  int i,j;
    int tx = item_ct1.get_local_id(2);

  int array_offset = offset*matrix_dim+offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
    array_offset += matrix_dim;
  }

    item_ct1.barrier(sycl::access::fence_space::local_space);

  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (tx>i){
      for(j=0; j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
      shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    if (tx>i){

      for(j=0; j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }

        item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
    array_offset += matrix_dim;
  }
}

SYCL_EXTERNAL void lud_perimeter(float *m, const int matrix_dim,
                                 const int offset, sycl::nd_item<3> item_ct1,
                                 float *dia, float *peri_row, float *peri_col) {

  int i,j, array_offset;
  int idx;

    int bx = item_ct1.get_group(2);
    int tx = item_ct1.get_local_id(2);

  if (tx < BLOCK_SIZE) {
    idx = tx;
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = tx-BLOCK_SIZE;

    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }

  }
    item_ct1.barrier(sycl::access::fence_space::local_space);

  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
    }
  }

    item_ct1.barrier(sycl::access::fence_space::local_space);

  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  }
}

SYCL_EXTERNAL void lud_internal(float *m, const int matrix_dim,
                                const int offset, sycl::nd_item<3> item_ct1,
                                float *peri_row, float *peri_col) {

    int bx = item_ct1.get_group(2);
    int by = item_ct1.get_group(1);

    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);

  int i;
  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;

  peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
  peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

    item_ct1.barrier(sycl::access::fence_space::local_space);

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
  m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;

}
