void gcrs_m_1_w_4_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 4;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result;
}

void gcrs_m_1_w_5_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 5;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result;
}

void gcrs_m_1_w_6_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 6;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result;
}

void gcrs_m_1_w_7_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 7;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result;
}

void gcrs_m_1_w_8_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 8;
  int i,j;
  long result = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result;
}

void gcrs_m_2_w_4_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 4;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

void gcrs_m_2_w_5_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 5;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

void gcrs_m_2_w_6_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 6;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

void gcrs_m_2_w_7_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 7;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

void gcrs_m_2_w_8_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 8;
  int i,j;
  long result[2];

  result[0] = 0;
  result[1] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
}

void gcrs_m_3_w_4_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 4;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

void gcrs_m_3_w_5_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 5;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

void gcrs_m_3_w_6_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 6;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

void gcrs_m_3_w_7_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 7;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

void gcrs_m_3_w_8_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 8;

  int i,j;
  long result[3];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
}

void gcrs_m_4_w_4_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 4;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

void gcrs_m_4_w_5_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 5;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

void gcrs_m_4_w_6_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 6;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

void gcrs_m_4_w_7_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
  long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 7;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);

  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

void gcrs_m_4_w_8_coding_dotprod(
  int k, int index, 
  const long *__restrict in, 
        long *__restrict out, 
  const unsigned int *__restrict bm, 
        long *__restrict shared_data, 
  nd_item<1> &item,
  int size)
{
  int w = 8;
  int i,j;
  long result[4];

  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

  int worksize_perblock = item.get_local_range(0) / w * w;
  const int tid = item.get_local_id(0);
  const unsigned int idx = worksize_perblock * item.get_group(0) + tid;

  if (tid >= worksize_perblock) {
    return;
  }

  if (idx >= size) {
    return;
  }

  int group_offset = (tid / w) * w;
  int group_inner_offset = tid % w;
  // row for each thread in the bitmatrix * row size which is k * w

  unsigned int bitInt = 0x01;
  unsigned int matrixInt;

  for ( i = 0; i < k; i++ ) {

    shared_data[tid] = *(in + i*size + idx);

    item.barrier(access::fence_space::local_space);

#pragma unroll
    for ( j = 0; j < w; j++ ) {
      matrixInt = bm[index];
      result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
      result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
      result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
      result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

      ++index;
    }
    item.barrier(access::fence_space::local_space);
  }

  out[idx] = result[0];
  out[idx + size] = result[1];
  out[idx + 2 * size] = result[2];
  out[idx + 3 * size] = result[3];
}

void m_1_w_4_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m1_w4_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_1_w_4_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_1_w_5_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m1_w5_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_1_w_5_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_1_w_6_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m1_w6_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_1_w_6_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_1_w_7_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m1_w7_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_1_w_7_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_1_w_8_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m1_w8_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_1_w_8_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}


void m_2_w_4_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m2_w4_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_2_w_4_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_2_w_5_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m2_w5_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_2_w_5_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_2_w_6_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m2_w6_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_2_w_6_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_2_w_7_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m2_w7_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_2_w_7_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}


void m_2_w_8_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m2_w8_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_2_w_8_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_3_w_4_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m3_w4_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_3_w_4_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_3_w_5_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m3_w5_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_3_w_5_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_3_w_6_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m3_w6_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_3_w_6_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_3_w_7_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m3_w7_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_3_w_7_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_3_w_8_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m3_w8_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_3_w_8_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_4_w_4_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m4_w4_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_4_w_4_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_4_w_5_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m4_w5_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_4_w_5_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_4_w_6_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m4_w6_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_4_w_6_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_4_w_7_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m4_w7_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_4_w_7_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void m_4_w_8_coding(
    queue &q, 
    int k, int index,
    buffer<char, 1> &d_data, 
    buffer<char, 1> &d_code,
    buffer<unsigned int, 1> &d_bitmatrix,
    int threadDimX,
    int blockDimX,
    int workSizePerGridInLong)
{
  range<1> gws (blockDimX * threadDimX);
  range<1> lws (threadDimX);
  
  auto d_data_re = d_data.reinterpret<long>(range<1>(d_data.size()/sizeof(long)));
  auto d_code_re = d_code.reinterpret<long>(range<1>(d_code.size()/sizeof(long)));
  
  q.submit([&] (handler &cgh) {
    auto d = d_data_re.get_access<sycl_read>(cgh); 
    auto c = d_code_re.get_access<sycl_discard_write>(cgh); 
    auto bm = d_bitmatrix.get_access<sycl_read>(cgh); 
    accessor<long, 1, sycl_read_write, access::target::local> sm (threadDimX, cgh);
    cgh.parallel_for<class m4_w8_dp>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      gcrs_m_4_w_8_coding_dotprod(k, 
                                  index,
                                  d.get_pointer(),
                                  c.get_pointer(),
                                  bm.get_pointer(),
                                  sm.get_pointer(),
                                  item,
                                  workSizePerGridInLong);
    });
  });
}

void (*coding_func_array[])(
    queue &q,
    int k, int index,
    buffer<char, 1> &dataPtr, 
    buffer<char, 1> &codeDevPtr,
    buffer<unsigned int, 1> &bitMatrixPtr, 
    int threadDimX,int blockDimX,
    int workSizePerGridInLong) = {
  m_1_w_4_coding,m_1_w_5_coding,m_1_w_6_coding,m_1_w_7_coding,m_1_w_8_coding,
  m_2_w_4_coding,m_2_w_5_coding,m_2_w_6_coding,m_2_w_7_coding,m_2_w_8_coding,
  m_3_w_4_coding,m_3_w_5_coding,m_3_w_6_coding,m_3_w_7_coding,m_3_w_8_coding,
  m_4_w_4_coding,m_4_w_5_coding,m_4_w_6_coding,m_4_w_7_coding, m_4_w_8_coding
};

