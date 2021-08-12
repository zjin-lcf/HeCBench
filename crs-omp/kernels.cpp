void m_1_w_4_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {


      int w = 4;
      int i,j;
      long result = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result;
      }
    }
  }
}

void m_1_w_5_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 5;
      int i,j;
      long result = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
          }
#pragma omp barrier
        }

        out[idx] = result;
      }
    }
  }
}

void m_1_w_6_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 6;
      int i,j;
      long result = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result;
      }
    }
  }
}

void m_1_w_7_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 7;
      int i,j;
      long result = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result;
      }
    }
  }
}

void m_1_w_8_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 8;
      int i,j;
      long result = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result;
      }
    }
  }
}


void m_2_w_4_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 4;
      int i,j;
      long result[2];

      result[0] = 0;
      result[1] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
      }
    }
  }
}

void m_2_w_5_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 5;
      int i,j;
      long result[2];

      result[0] = 0;
      result[1] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
      }
    }
  }
}

void m_2_w_6_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 6;
      int i,j;
      long result[2];

      result[0] = 0;
      result[1] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
      }
    }
  }
}

void m_2_w_7_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 7;
      int i,j;
      long result[2];

      result[0] = 0;
      result[1] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
      }
    }
  }
}


void m_2_w_8_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {
      int w = 8;
      int i,j;
      long result[2];

      result[0] = 0;
      result[1] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
      }
    }
  }
}

void m_3_w_4_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 4;

      int i,j;
      long result[3];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier
        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
      }
    }
  }
}

void m_3_w_5_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 5;

      int i,j;
      long result[3];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];

      }
    }
  }
}

void m_3_w_6_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 6;

      int i,j;
      long result[3];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
      }
    }
  }
}

void m_3_w_7_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 7;

      int i,j;
      long result[3];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
      }
    }
  }
}

void m_3_w_8_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 8;

      int i,j;
      long result[3];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
      }
    }
  }
}

void m_4_w_4_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {


      int w = 4;
      int i,j;
      long result[4];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;
      result[3] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
        out[idx + 3 * size] = result[3];
      }
    }
  }
}

void m_4_w_5_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 5;
      int i,j;
      long result[4];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;
      result[3] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {


        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
        out[idx + 3 * size] = result[3];
      }
    }
  }
}

void m_4_w_6_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 6;
      int i,j;
      long result[4];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;
      result[3] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {


        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
        out[idx + 3 * size] = result[3];
      }
    }
  }
}

void m_4_w_7_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 7;
      int i,j;
      long result[4];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;
      result[3] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
        out[idx + 3 * size] = result[3];
      }
    }
  }
}

void m_4_w_8_coding(int k, int index,
    const long *in, 
    long *out,
    const unsigned int *bm,
    int threadDimX,int blockDimX,
    int size)
{
#pragma omp target teams num_teams(blockDimX) thread_limit(threadDimX)
  {
    long shared_data[128];
#pragma omp parallel 
    {

      int w = 8;
      int i,j;
      long result[4];

      result[0] = 0;
      result[1] = 0;
      result[2] = 0;
      result[3] = 0;

      const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

      const int tid = omp_get_thread_num();
      const int gid = omp_get_team_num();
      int worksize_perblock = omp_get_num_threads() / w * w;
      const unsigned int idx = worksize_perblock * gid + tid;

      if (tid < worksize_perblock && idx < size) {

        int group_offset = (tid / w) * w;
        int group_inner_offset = tid % w;
        // row for each thread in the bitmatrix * row size which is k * w

        unsigned int bitInt = 0x01;
        unsigned int matrixInt;

        for ( i = 0; i < k; i++ ) {

          shared_data[tid] = *(in + i*size + idx);

#pragma omp barrier

#pragma unroll
          for ( j = 0; j < w; j++ ) {
            matrixInt = bm[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
          }
#pragma omp barrier

        }

        out[idx] = result[0];
        out[idx + size] = result[1];
        out[idx + 2 * size] = result[2];
        out[idx + 3 * size] = result[3];
      }
    }
  }
}

void (*coding_func_array[])(int k, int index,
    const long *dataPtr, long *codeDevPtr,
    const unsigned int *bitMatrixPtr, 
    int threadDimX,int blockDimX,
    int workSizePerGridInLong) = {
  m_1_w_4_coding,m_1_w_5_coding,m_1_w_6_coding,m_1_w_7_coding,m_1_w_8_coding,
  m_2_w_4_coding,m_2_w_5_coding,m_2_w_6_coding,m_2_w_7_coding,m_2_w_8_coding,
  m_3_w_4_coding,m_3_w_5_coding,m_3_w_6_coding,m_3_w_7_coding,m_3_w_8_coding,
  m_4_w_4_coding,m_4_w_5_coding,m_4_w_6_coding,m_4_w_7_coding,m_4_w_8_coding
};

