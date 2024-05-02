#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

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

  hipError_t hipStat;
  hipblasStatus_t stat;
  hipblasHandle_t handle;

  stat = hipblasCreate(&handle);
  if(stat != HIPBLAS_STATUS_SUCCESS){
    cerr << "hipblas init failed" << endl;
    exit(1);
  }

  if(verbose) cout << "allocating device variables" << endl;

  // allocate input space on device
  float *devMatrices;
  hipStat = hipMalloc((void**)&devMatrices, matrices_size);
  assert(!hipStat);

  float *devVectors;
  hipStat = hipMalloc((void**)&devVectors, vectors_size);
  assert(!hipStat);

  // allocate result space on device
  float *devResult;
  hipStat = hipMalloc((void**)&devResult, vectors_size);

  assert(!hipStat);

  if(verbose) cout << "copying data to device" << endl;
  // copy data to device
  hipStat = 
    hipMemcpy(devMatrices, matrices, matrices_size, hipMemcpyHostToDevice);

  assert(!hipStat);
  
  hipStat = 
    hipMemcpy(devVectors, vectors, vectors_size, hipMemcpyHostToDevice);

  assert(!hipStat);

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
  hipStat = hipMalloc((void**)&devAList, num * sizeof(float*));
  assert(!hipStat);

  hipStat = hipMalloc((void**)&devBList, num * sizeof(float*));
  assert(!hipStat);

  hipStat = hipMalloc((void**)&devCList, num * sizeof(float*));
  assert(!hipStat);

  hipStat = hipMemcpy(devAList, AList, num * sizeof(float*), hipMemcpyHostToDevice);
  assert(!hipStat);
  
  hipStat = hipMemcpy(devBList, BList, num * sizeof(float*), hipMemcpyHostToDevice);
  assert(!hipStat);

  hipStat = hipMemcpy(devCList, CList, num * sizeof(float*), hipMemcpyHostToDevice);
  assert(!hipStat);


  /* perform <num> <size x size> x <size x 1> multiplications 
     with distinct matrices
   */
  for(int size = lower; size <= upper; size++){
    if(verbose) cout << "running with <size x size> x <size x 1> " << size << endl;
    double sum = 0.0;
    const int m = size, n = 1, k = size;
    for(int rep = 0; rep <= reps; rep++){
      auto start = std::chrono::steady_clock::now();
      stat = hipblasSgemmBatched(handle,
                                 HIPBLAS_OP_N,
                                 HIPBLAS_OP_N,
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
      hipDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      auto elapsed = time * 1e-3;

      if(stat != HIPBLAS_STATUS_SUCCESS){
	cerr << "hipblasSgemmBatched failed" << endl;
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

  hipFree(devMatrices);
  hipFree(devVectors);
  hipFree(devResult);
  hipFree(devAList);
  hipFree(devBList);
  hipFree(devCList);

  free(matrices);
  free(vectors);
  free(AList);
  free(BList);
  free(CList);
      
  return 0;
}
