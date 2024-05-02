#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

int main(int argc, char ** argv){

  int status;
  int lower = 2;    // lower bound to the matrix dimension
  int upper = 100;  // upper bound to the matrix dimension
  int num = 25000;  // batch size
  int reps = 10;
  int verbose = 0;
  
  while((status = getopt(argc, argv, "l:u:n:r:v")) != -1){
    switch(status){
    case 'l':
      lower = strtoul(optarg, 0, 0);
      break;
    case 'u':
      upper = strtoul(optarg, 0, 0);
      break;
    case 'n':
      num = strtoul(optarg, 0, 0);  // batch size
      break;
    case 'r':
      reps = strtoul(optarg, 0, 0);
      break;
    case 'v':
      verbose = 1;
      break;
    default:
      cerr << "invalid argument: " << status << endl;
      exit(1);
    }
  }

  cout << "running with" << " lower: " << lower << " upper: " << upper
       << " num: " << num << " reps: " << reps << endl;

  if(verbose) cout << "initializing inputs" << endl;
  size_t matrices_size = upper * upper * num * sizeof(float);
  size_t vectors_size = upper * num * sizeof(float);

  float *matrices = (float*)malloc(matrices_size);
  assert(matrices);

  float *vectors = (float*)malloc(vectors_size);
  assert(vectors);

  srand48(48);
  for(int i = 0; i < num * upper * upper; i++)
    matrices[i] = drand48();

  for(int i = 0; i < num * upper; i++)
    vectors[i] = drand48();

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if(stat != CUBLAS_STATUS_SUCCESS){
    cerr << "cublas init failed" << endl;
    exit(1);
  }

  if(verbose) cout << "allocating device variables" << endl;

  // allocate input space on device
  float *devMatrices;
  cudaStat = cudaMalloc((void**)&devMatrices, matrices_size);
  assert(!cudaStat);

  float *devVectors;
  cudaStat = cudaMalloc((void**)&devVectors, vectors_size);
  assert(!cudaStat);

  // allocate result space on device
  float *devResult;
  cudaStat = cudaMalloc((void**)&devResult, vectors_size);

  assert(!cudaStat);

  if(verbose) cout << "copying data to device" << endl;
  // copy data to device
  cudaStat = 
    cudaMemcpy(devMatrices, matrices, matrices_size, cudaMemcpyHostToDevice);

  assert(!cudaStat);
  
  cudaStat = 
    cudaMemcpy(devVectors, vectors, vectors_size, cudaMemcpyHostToDevice);

  assert(!cudaStat);

  // create lists of device pointers to inputs and outputs
  float **AList = 0, **BList = 0, **CList = 0;

  AList = (float**)malloc(num * sizeof(float*));
  BList = (float**)malloc(num * sizeof(float*));
  CList = (float**)malloc(num * sizeof(float*));

  int lda = upper, // lda >= max(1,m)
      ldb = upper, // ldb >= max(1,k)
      ldc = upper; // ldc >= max(1,m)

  const float alpha = 1.0f, beta = 0.0f;
  for(int i = 0; i < num; i++){
    // each array of dim. lda x k
    AList[i] = devMatrices + upper * upper * i;
    // each array of dim. ldb x n
    BList[i] = devVectors + upper * i;
    // each array of dim. ldc x n
    CList[i] = devResult + upper * i;
  }

  // copy pointer lists to device
  float **devAList, **devBList, **devCList;
  cudaStat = cudaMalloc((void**)&devAList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMalloc((void**)&devBList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMalloc((void**)&devCList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMemcpy(devAList, AList, num * sizeof(float*), cudaMemcpyHostToDevice);
  assert(!cudaStat);
  
  cudaStat = cudaMemcpy(devBList, BList, num * sizeof(float*), cudaMemcpyHostToDevice);
  assert(!cudaStat);

  cudaStat = cudaMemcpy(devCList, CList, num * sizeof(float*), cudaMemcpyHostToDevice);
  assert(!cudaStat);


  /* perform <num> <size x size> x <size x 1> multiplications 
     with distinct matrices
   */
  for(int size = lower; size <= upper; size++){
    if(verbose) cout << "running with <size x size> x <size x 1> " << size << endl;
    double sum = 0.0;
    const int m = size, n = 1, k = size;
    for(int rep = 0; rep <= reps; rep++){
      auto start = std::chrono::steady_clock::now();
      stat = cublasSgemmBatched(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                m, n, k,
                                &alpha,
                                (const float**)devAList,
                                lda,
                                (const float**)devBList,
                                ldb,
                                &beta,
                                devCList,
                                ldc,
                                num);
      cudaDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      auto elapsed = time * 1e-3;

      if(stat != CUBLAS_STATUS_SUCCESS){
	cerr << "cublasSgemmBatched failed" << endl;
        break;
      }

      if (rep != 0) sum += elapsed;
      
      if(verbose)
	cout << "size " << size << ": " << elapsed << " us; " 
	     << elapsed / num << " us per operation" << endl;
    }
    cout << "size " << size << " average execution time: " << sum/reps << " us; "
	 << sum / reps / num << " us per operation" << endl;
  }

  cudaFree(devMatrices);
  cudaFree(devVectors);
  cudaFree(devResult);
  cudaFree(devAList);
  cudaFree(devBList);
  cudaFree(devCList);

  free(matrices);
  free(vectors);
  free(AList);
  free(BList);
  free(CList);
      
  return 0;
}
