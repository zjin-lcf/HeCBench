/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <stdio.h>
#include <algorithm> 
#include <iostream> 
#include <vector> 

using namespace std; 

#define MAX_KERNEL_THREADS 256

// float or double 
typedef float vtype;
typedef vector<vector<vtype>> matrix; 

template<typename T>
__device__
T parallel_prefix_sum(const int n, const int *ind, const T *w) {
  int i,j,mn;
  T v,last;
  T sum=0.0;
  bool valid;

  mn =(((n+blockDim.x-1)/blockDim.x)*blockDim.x); //n in multiple of blockDim.x
  for (i=threadIdx.x; i<mn; i+=blockDim.x) {
    //All threads (especially the last one) must always participate
    //in the shfl instruction, otherwise their sum will be undefined.
    //So, the loop stopping condition is based on multiple of n in loop increments,
    //so that all threads enter into the loop and inside we make sure we do not
    //read out of bounds memory checking for the actual size n.

    //check if the thread is valid
    valid  = i<n;

    //Notice that the last thread is used to propagate the prefix sum.
    //For all the threads, in the first iteration the last is 0, in the following
    //iterations it is the value at the last thread of the previous iterations.

    //get the value of the last thread
    last = __shfl(sum, blockDim.x-1, blockDim.x);

    //if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[i]] : 0.0;

    //do prefix sum (of size warpSize=blockDim.x =< 32)
    for (j=1; j<blockDim.x; j*=2) {
      v = __shfl_up(sum, j, blockDim.x);
      if (threadIdx.x >= j) sum+=v;
    }
    //shift by last
    sum+=last;
    //notice that no __threadfence or __syncthreads are needed in this implementation
  }
  //get the value of the last thread (to all threads)
  last = __shfl(sum, blockDim.x-1, blockDim.x);

  return last;
}

// Volume of neighboors (*weight_s)
template<bool weighted, typename T>
__global__ void 
jaccard_row_sum(const int n, const int *csrPtr, const int *csrInd, const T *w, T *work) {
  int row,start,end,length;
  T sum;

  for (row=threadIdx.y+blockIdx.y*blockDim.y; row<n; row+=gridDim.y*blockDim.y) {
    start = csrPtr[row];
    end   = csrPtr[row+1];
    length= end-start;
    //compute row sums 
    if (weighted) {
      sum = parallel_prefix_sum(length, csrInd + start, w); 
      if (threadIdx.x == 0) work[row] = sum;
    }
    else {
      work[row] = (T)length;
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template<bool weighted, typename T>
__global__ void 
jaccard_is(const int n, const int e, const int *csrPtr, const int *csrInd, 
    const T *v, const T *work, T *weight_i, T *weight_s) {

  for (int row=threadIdx.z+blockIdx.z*blockDim.z; row<n; row+=gridDim.z*blockDim.z) {  
    for (int j=csrPtr[row]+threadIdx.y+blockIdx.y*blockDim.y; j<csrPtr[row+1]; j+=gridDim.y*blockDim.y) { 
      int col = csrInd[j];
      //find which row has least elements (and call it reference row)
      int Ni = csrPtr[row+1] - csrPtr[row];
      int Nj = csrPtr[col+1] - csrPtr[col];
      int ref= (Ni < Nj) ? row : col;
      int cur= (Ni < Nj) ? col : row;

      //compute new sum weights
      weight_s[j] = work[row] + work[col];

      //compute new intersection weights 
      //search for the element with the same column index in the reference row
      for (int i=csrPtr[ref]+threadIdx.x+blockIdx.x*blockDim.x; i<csrPtr[ref+1]; i+=gridDim.x*blockDim.x) {
        int match  =-1;           
        int ref_col = csrInd[i];
        T ref_val = weighted ? v[ref_col] : (T)1.0;

        //binary search (column indices are sorted within each row)
        int left = csrPtr[cur]; 
        int right= csrPtr[cur+1]-1; 
        while(left <= right){
          int middle = (left+right)>>1; 
          int cur_col= csrInd[middle];
          if (cur_col > ref_col) {
            right=middle-1;
          }
          else if (cur_col < ref_col) {
            left=middle+1;
          }
          else {
            match = middle; 
            break; 
          }
        }            

        //if the element with the same column index in the reference row has been found
        if (match != -1){
          atomicAdd(&weight_i[j],ref_val);
        }
      }
    }
  }
}

template<bool weighted, typename T>
__global__ void 
jaccard_jw(const int e, 
		const T *csrVal, 
		const T gamma, 
		const T *weight_i, 
		const T *weight_s, 
		T *weight_j) 
{
  for (int j=threadIdx.x+blockIdx.x*blockDim.x; j<e; j+=gridDim.x*blockDim.x) {  
    T Wi =  weight_i[j];
    T Ws =  weight_s[j];
    weight_j[j] = (gamma*csrVal[j])* (Wi/(Ws-Wi));
  }
}



template <bool weighted, typename T>
__global__ void 
fill(const int e, T* w, const T value) {
  for (int j=threadIdx.x+blockIdx.x*blockDim.x; j<e; j+=gridDim.x*blockDim.x) {  
    w[j] = weighted ? (T)(j+1)/e : value; // non-zeron weights when weighted 
  }
}

template <bool weighted, typename T>
void jaccard_weight (const int n, const int e, 
    int* csr_ptr, int* csr_ind, T* csr_val)
{

  const T gamma = (T)0.46;  // arbitrary

  T *weight_i, *weight_s, *weight_j, *work;
  int* csrInd;
  int* csrPtr;
    T* csrVal;

  // dump 
  T* wj = (T*) malloc (sizeof(T) * e);
#ifdef DEBUG
  T* wi = (T*) malloc (sizeof(T) * e);
  T* ws = (T*) malloc (sizeof(T) * e);
#endif

  cudaMalloc ((void**)&work, sizeof(T) * n);
  cudaMalloc ((void**)&weight_i, sizeof(T) * e);
  cudaMalloc ((void**)&weight_s, sizeof(T) * e);

  // initialize weight
  cudaMalloc ((void**)&weight_j, sizeof(T) * e);
  fill<weighted, T><<<(e+255)/256, 256>>>(e, weight_j, (T)1.0);

  cudaMalloc ((void**)&csrPtr, sizeof(int) * (n+1));
  cudaMemcpyAsync(csrPtr, csr_ptr, sizeof(int) * (n+1), cudaMemcpyHostToDevice, 0);

  cudaMalloc ((void**)&csrInd, sizeof(int) * e);
  cudaMemcpyAsync(csrInd, csr_ind, sizeof(int) * e, cudaMemcpyHostToDevice, 0);

  cudaMalloc ((void**)&csrVal, sizeof(T) * e);
  cudaMemcpyAsync(csrVal, csr_val, sizeof(T) * e, cudaMemcpyHostToDevice, 0);

  dim3 nthreads, nblocks;

  const int y=4;

  // compute row sum with prefix sum
  nthreads.x = 32/y; 
  nthreads.y = y; 
  nthreads.z = 1; 
  nblocks.x  = 1; 
  nblocks.y  = (n + nthreads.y - 1)/nthreads.y;  // less than MAX CUDA BLOCKs
  nblocks.z  = 1; 
  jaccard_row_sum<weighted,T><<<nblocks,nthreads>>>(n,csrPtr,csrInd,weight_j,work);

  // initialize volume of intersections
  fill<false, T><<<(e+255)/256, 256>>>(e, weight_i, (T)0.0);

  // compute volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
  nthreads.x = 32/y;
  nthreads.y = y;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = (n + nthreads.z - 1)/nthreads.z; // less than CUDA_MAX_BLOCKS);
  jaccard_is<weighted,T><<<nblocks,nthreads>>>(n,e,csrPtr,csrInd,weight_j,work,weight_i,weight_s);

#ifdef DEBUG
  cudaMemcpy(wi, weight_i, sizeof(T) * e, cudaMemcpyDeviceToHost);
  cudaMemcpy(ws, weight_s, sizeof(T) * e, cudaMemcpyDeviceToHost);
  for (int i = 0; i < e; i++) printf("wi: %d %f\n", i, wi[i]);
  for (int i = 0; i < e; i++) printf("ws: %d %f\n", i, ws[i]);
#endif

  // compute jaccard weights
  nthreads.x = std::min(e, MAX_KERNEL_THREADS); 
  nthreads.y = 1; 
  nthreads.z = 1;  
  nblocks.x  = (e + nthreads.x - 1)/nthreads.x;  // less than MAX CUDA BLOCKs
  nblocks.y  = 1; 
  nblocks.z  = 1;
  jaccard_jw<weighted,T><<<nblocks,nthreads>>>(e,csrVal,gamma,weight_i,weight_s,weight_j);


  cudaMemcpy(wj, weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost);
#ifdef DEBUG
  for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, wj[i]);
#endif

  cudaFree (work);
  cudaFree (weight_i);
  cudaFree (weight_s);
  cudaFree (weight_j);
  cudaFree (csrInd);
  cudaFree (csrVal);
  cudaFree (csrPtr);
  free(wj);
#ifdef DEBUG
  free(wi);
  free(ws);
#endif
}

// Utilities
void printMatrix(const matrix& M) 
{ 
  int m = M.size(); 
  int n = M[0].size(); 
  for (int i = 0; i < m; i++) { 
    for (int j = 0; j < n; j++) 
      cout << M[i][j] << " ";     
    cout << endl; 
  } 
} 

template <typename T>
void printVector(const vector<T>& V, char* msg) 
{ 
  cout << msg << "[ "; 
  for_each(V.begin(), V.end(), [](int a) { cout << a << " "; }); 
  cout << "]" << endl; 
} 

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv) 
{ 
  int iter = 1;  

#ifdef DEBUG
  matrix M  = { 
    { 0, 0, 0, 0, 1 }, 
    { 5, 8, 0, 0, 0 }, 
    { 0, 0, 3, 0, 0 }, 
    { 0, 6, 0, 0, 1 } 
  }; 
#else
  matrix M;
  int numRow = atoi(argv[1]);
  int numCol = atoi(argv[2]);
  iter = atoi(argv[3]);
  vector<vtype> rowElems(numCol);
  for (int r = 0; r < numRow; r++) {
    for (int c = 0; c < numCol; c++)
      rowElems[c] = rand() % 10;
    M.push_back(rowElems);
  }
#endif

  int row = M.size();
  int col = M[0].size();
  printf("Number of matrix rows and cols: %d %d\n", row, col);
  vector<vtype> csr_val;
  vector<int> csr_ptr = { 0 }; // add -std=c++11  
  vector<int> csr_ind;
  int NNZ = 0; 

  for (int i = 0; i < row; i++) { 
    for (int j = 0; j < col; j++) { 
      if (M[i][j] != (vtype)0) { 
        csr_val.push_back(M[i][j]); 
        csr_ind.push_back(j); 
        NNZ++; // count Number of Non Zero Elements in row i 
      } 
    } 
    csr_ptr.push_back(NNZ); 
  } 

  if (row <= 16 && col <= 16) {
    printMatrix(M); 
    printVector(csr_val, (char*)"values = "); 
    printVector(csr_ptr, (char*)"row pointer = "); 
    printVector(csr_ind, (char*)"col indices = "); 
  }

  for (int i = 0; i < iter; i++) {
    jaccard_weight<true, vtype>(row, col, csr_ptr.data(), csr_ind.data(), csr_val.data());
    jaccard_weight<false, vtype>(row, col, csr_ptr.data(), csr_ind.data(), csr_val.data());
  }

  return 0; 
} 

