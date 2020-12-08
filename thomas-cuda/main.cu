#include <iostream>
#include "ThomasMatrix.hpp"
#include "utils.hpp"
#include "cuThomasBatch.h"

// CPU kernel
void solve_seq(const double* l, const double* d, double* u, double* rhs, const int n, const int N) 
{
  int first,last;
  for (int j = 0; j < N; ++j)
  {
    first = j*n;
    last = first + n;
    last--; 

    u[first] /= d[first];
    rhs[first] /= d[first];

    for (int i = first+1; i < last; i++) {
      u[i] /= d[i] - l[i]*u[i-1];
      rhs[i] = (rhs[i] - l[i]*rhs[i-1]) / (d[i] - l[i]*u[i-1]);
    }

    rhs[last] = (rhs[last] - l[last]*rhs[last-1]) / (d[last] - l[last]*u[last-1]);

    for (int i = last-1; i >= first; i--) {
      rhs[i] -= u[i]*rhs[i+1];
    }
  }
}

int main(int argc, char const *argv[])
{

  if(argc < 4 or argc > 4){
    std::cout << "Usage: ./run [system size] [#systems] [thread block size]" << std::endl;
    return -1;
  }

  int M = std::stoi(argv[1]); // c++11
  int N = std::stoi(argv[2]);
  int BlockSize  = std::stoi(argv[3]);  // GPU thread block size

  //Loading a synthetic tridiagonal matrix into our structure
  ThomasMatrix params = loadThomasMatrixSyn(M);

  double* u_seq = (double*) malloc(N*params.M*sizeof(double));
  double* u_Thomas_host =  (double*) malloc(N*params.M*sizeof(double));
  double* u_input = (double*) malloc(N*params.M*sizeof(double));

  double* d_seq = (double*) malloc(N*params.M*sizeof(double));
  double* d_Thomas_host =  (double*) malloc(N*params.M*sizeof(double));
  double* d_input = (double*) malloc(N*params.M*sizeof(double));

  double* l_seq = (double*) malloc(N*params.M*sizeof(double));
  double* l_Thomas_host =  (double*) malloc(N*params.M*sizeof(double));
  double* l_input = (double*) malloc(N*params.M*sizeof(double));

  double* rhs_seq = (double*) malloc(N*params.M*sizeof(double));
  double* rhs_Thomas_host = (double*) malloc(N*params.M*sizeof(double));
  double* rhs_input = (double*) malloc(N*params.M*sizeof(double));

  double* rhs_seq_output = (double*) malloc(N*params.M*sizeof(double));
  double* rhs_Thomas_output=(double*) malloc(N*params.M*sizeof(double));
  double* rhs_seq_interleave = (double*) malloc(N*params.M*sizeof(double));

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < params.M; ++j)
    {
      u_seq[(i * params.M) + j] = params.a[j];
      u_input[(i * params.M) + j] = params.a[j];

      d_seq[(i * params.M) + j] = params.d[j];
      d_input[(i * params.M) + j] = params.d[j];

      l_seq[(i * params.M) + j] = params.b[j];
      l_input[(i * params.M) + j] = params.b[j];

      rhs_seq[(i * params.M) + j] = params.rhs[j];
      rhs_input[(i * params.M) + j] = params.rhs[j];

    }
  }


  // Sequantial CPU Execution for correct error check
  double init = time_wtime();
  solve_seq( l_seq, d_seq, u_seq, rhs_seq, params.M, N );
  printf("        CPU SEQ Time(s) %e\n", time_wtime()-init);

  for (int i = 0; i < params.M*N; ++i) {
    rhs_seq_output[i] = rhs_seq[i];
    //printf("%f\n", rhs_seq[i]);
  }

  // initialize again because u_seq and rhs_seq are modified by solve_seq
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < params.M; ++j)
    {
      u_seq[(i * params.M) + j] = params.a[j];
      u_input[(i * params.M) + j] = params.a[j];

      d_seq[(i * params.M) + j] = params.d[j];
      d_input[(i * params.M) + j] = params.d[j];

      l_seq[(i * params.M) + j] = params.b[j];
      l_input[(i * params.M) + j] = params.b[j];

      rhs_seq[(i * params.M) + j] = params.rhs[j];
      rhs_input[(i * params.M) + j] = params.rhs[j];

    }
  }


  // transpose the inputs for sequential accesses on a GPU 
  for (int i = 0; i < params.M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      u_Thomas_host[i*N+j] = u_input[j*params.M+i];
      l_Thomas_host[i*N+j] = l_input[j*params.M+i];
      d_Thomas_host[i*N+j] = d_input[j*params.M+i];
      rhs_Thomas_host[i*N+j] = rhs_input[j*params.M+i];
      rhs_seq_interleave[i*N+j] = rhs_seq_output[j*params.M+i];

    }
  }

 
  // Run GPU kernel

  double *u_device;
  double *d_device;
  double *l_device;
  double *rhs_device;

  cudaMalloc((void**)&u_device ,N*params.M*sizeof(double));
  cudaMalloc((void**)&d_device ,N*params.M*sizeof(double));
  cudaMalloc((void**)&l_device ,N*params.M*sizeof(double));
  cudaMalloc((void**)&rhs_device ,N*params.M*sizeof(double));

  init = time_wtime();
  for (int n = 0; n < 100; n++) {
    cudaMemcpyAsync(u_device,u_Thomas_host,N*params.M*sizeof(double),cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(l_device,l_Thomas_host,N*params.M*sizeof(double),cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_device,d_Thomas_host,N*params.M*sizeof(double),cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(rhs_device,rhs_Thomas_host,N*params.M*sizeof(double),cudaMemcpyHostToDevice, 0);
    cuThomasBatch<<<(N/BlockSize)+1, BlockSize>>> (l_device, d_device, u_device, rhs_device, params.M, N);
    cudaMemcpyAsync(rhs_Thomas_output,rhs_device,N*params.M*sizeof(double),cudaMemcpyDeviceToHost, 0);
  }
  cudaDeviceSynchronize();
  printf("        cuThomasBatch Time(s) %e   ", (time_wtime()-init)/100);

  // verify
  calcError(rhs_seq_interleave,rhs_Thomas_output,N*params.M);


  free(u_seq);  
  free(u_Thomas_host);
  free(u_input);

  free(d_seq);  
  free(d_Thomas_host);
  free(d_input);

  free(l_seq);  
  free(l_Thomas_host);
  free(l_input);

  free(rhs_seq);  
  free(rhs_Thomas_host);
  free(rhs_input);

  free(rhs_seq_output);
  free(rhs_Thomas_output);
  free(rhs_seq_interleave);

  cudaFree(l_device);
  cudaFree(d_device);
  cudaFree(u_device);
  cudaFree(rhs_device);

  return 0;

}


