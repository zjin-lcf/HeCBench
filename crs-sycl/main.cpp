//  Created by Liu Chengjian on 17/10/9.
//  Copyright (c) 2017 csliu. All rights reserved.
//

#include <sys/time.h>
#include "GCRSMatrix.h"
#include "utils.h"
#include "common.h"

typedef void (*coding_func)(
    queue &q,
    int k, int index,
    buffer<char,1> &dataPtr,
    buffer<char,1> &codeDevPtr,
    buffer<unsigned int,1> &bitMatrixPtr,
    int threadDimX,int blockDimX,
    int workSizePerGridInLong);

#include "kernels.cpp"

int main(int argc, const char * argv[]) {

  if (argc != 3) {
    printf("Usage: ./%s workSizePerDataParityBlockInMB numberOfTasks\n", argv[0]);
    exit(0);
  }

  int bufSize = atoi(argv[1]) * 1024 * 1024; // workSize per data parity block
  int taskNum = atoi(argv[2]);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  double encode_time = 0.0;

#ifdef DUMP
  for (int m = 4; m <= 4; ++m) {
  for (int n = 8; n <= 8; ++n) {  // w is updated in the nested loop
  for (int k = MAX_K; k <= MAX_K; ++k) {
#else
  for (int m = 1; m <= 4; ++m) {
  for (int n = 4; n <= 8; ++n) {  // w is updated in the nested loop
  for (int k = m; k <= MAX_K; ++k) {
#endif

    int w = gcrs_check_k_m_w(k, m, n);
    if (w < 0) continue;

#ifdef DUMP
    printf("k:%d, m:%d w:%d\n",k,m,w);
#endif

    int *bitmatrix = gcrs_create_bitmatrix(k, m, w);
    //printMatrix(bitmatrix, k*w, m*w);

    //  adjust the bufSize
    int bufSizePerTask = align_value(bufSize / taskNum, sizeof(long) * w);
    bufSize = bufSizePerTask * taskNum;

    //  compute the bufSize for the last task
    int bufSizeForLastTask = bufSize - (bufSizePerTask * (taskNum - 1));
#ifdef DUMP
    printf("Total Size:%d Size per task:%d Size for last task:%d\n", 
           bufSize, bufSizePerTask, bufSizeForLastTask);
#endif

    // allocate host buffers
    char* data = (char*) malloc (bufSize * k);
    char* code = (char*) malloc (bufSize * m);

    // initialize host buffer
    generateRandomValue(data, bufSize * k);

    int dataSizePerAssign = bufSizePerTask * k;
    int codeSizePerAssign = bufSizePerTask * m;

    // taskSize will determine the number of kernels to run on a device
    int taskSize = 1;
    int mRemain = m;

    // adjust taskSize
    if (m >= MAX_M) {
      taskSize = m / MAX_M;
      if (m % MAX_M != 0) ++taskSize;
    }

#ifdef DUMP
    printf("task size: %d\n", taskSize);
#endif

    // set up kernel execution parameters
    int *mValue = (int*) malloc (sizeof(int) * taskSize);
    int *index = (int*) malloc (sizeof(int) * taskSize);
    coding_func *coding_function_ptrs = (coding_func*) malloc (sizeof(coding_func) * taskSize);

    for (int i = 0; i < taskSize; ++i) {
      if (mRemain < MAX_M) {
        mValue[i] = mRemain;
      }else{
        mValue[i] = MAX_M;
        mRemain = mRemain - MAX_M;
      }

      if (i == 0) {
        index[i] = 0;
      }else{
        index[i] = index[i-1] + k * w;
      }
      coding_function_ptrs[i] = coding_func_array[(mValue[i] - 1) * (MAX_W - MIN_W + 1)+ w - MIN_W];
    }

    //  create and then update encoding bit matrix
    unsigned int *all_columns_bitmatrix = (unsigned int*) malloc (sizeof(unsigned int) * k * w * taskSize);

    int mValueSum = 0;
    for (int i = 0; i < taskSize; ++i) {

      unsigned int *column_bitmatrix = gcrs_create_column_coding_bitmatrix(
          k, mValue[i], w, bitmatrix + k * w * mValueSum * w);

      memcpy((all_columns_bitmatrix + i * k * w), column_bitmatrix, k * w * sizeof(unsigned int));

      free(column_bitmatrix);
      mValueSum += mValue[i];
    }

    // allocate device buffers
    buffer<char, 1> d_data (bufSize * k);
    buffer<char, 1> d_code (bufSize * m);
    buffer<unsigned int, 1> d_bitmatrix (k * w * taskSize);

    q.submit([&] (handler &cgh) {
      auto acc = d_bitmatrix.get_access<sycl_discard_write>(cgh);
      cgh.copy(all_columns_bitmatrix, acc);
    }).wait();

    int warpThreadNum = 32;
    int threadNum = MAX_THREAD_NUM;
    size_t workSizePerWarp = warpThreadNum / w * w;
    size_t workSizePerBlock = threadNum / warpThreadNum * workSizePerWarp * sizeof(size_t);
    size_t blockNum = bufSizePerTask / workSizePerBlock;

    if ((bufSizePerTask % workSizePerBlock) != 0) {
      blockNum = blockNum + 1;
    }

#ifdef DUMP
    printf("#blocks: %zu  blockSize: %d\n", blockNum, threadNum);
#endif

    struct timeval startEncodeTime, endEncodeTime;
    gettimeofday(&startEncodeTime, NULL);

    for (int i = 0; i < taskNum; ++i) {

      int count = (i == taskNum-1) ? bufSizeForLastTask : bufSizePerTask;

      q.submit([&] (handler &cgh) {
        auto acc = d_data.get_access<sycl_write>(cgh, range<1>(k*count), id<1>(i*k*bufSizePerTask));
        cgh.copy(data + i * k * bufSizePerTask , acc);
      });

      int workSizePerGrid = count / sizeof(long);
      int size = workSizePerGrid * sizeof(long);
      mValueSum = 0;
      buffer<char, 1> d_data_sub (d_data, id<1>(dataSizePerAssign*i), range<1>(bufSize*k-dataSizePerAssign*i));
      for (int j = 0; j < taskSize; ++j) {
        buffer<char, 1> d_code_sub (d_code, 
		   id<1>(codeSizePerAssign*i+mValueSum*size), 
		   range<1>(bufSize*m - codeSizePerAssign*i+mValueSum*size));

        coding_function_ptrs[j](
          q, 
          k, index[j], 
          d_data_sub, 
          d_code_sub, 
          d_bitmatrix, 
          threadNum, blockNum, workSizePerGrid);

        mValueSum += mValue[j];
      }

      q.submit([&] (handler &cgh) {
        auto acc = d_code.get_access<sycl_read>(cgh, range<1>(m*count), id<1>(i*m*bufSizePerTask));
        cgh.copy(acc, code + i * m * bufSizePerTask);
      });
    }
    q.wait();
    gettimeofday(&endEncodeTime, NULL);
    double etime = elapsed_time_in_ms(startEncodeTime, endEncodeTime);
#ifdef DUMP
    printf("Encoding time over %d tasks: %lf (ms)\n", taskNum, etime);
#endif
    encode_time += etime;

#ifdef DUMP
    for (int i = 0; i < bufSize*m; i++) printf("%d\n", code[i]);
    printf("\n");
#endif

    free(mValue);
    free(index);
    free(coding_function_ptrs);
    free(bitmatrix);
    free(all_columns_bitmatrix);
    free(code);
    free(data);
  }
  }
  }

  printf("Total encoding time %lf (s)\n", encode_time * 1e-3);

  return 0;
}
