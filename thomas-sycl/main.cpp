#include <iostream>
#include "common.h"
#include "ThomasMatrix.hpp"
#include "utils.hpp"

// CPU kernel
void solve_seq(const double* l, const double* d, double* u, double* rhs, const int n, const int N) 
{
  int first,last;
  for (int j = 0; j < N; ++j)
  {
    first = j*n;
    last = first + n - 1;

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

  const int M = std::stoi(argv[1]); // c++11
  const int N = std::stoi(argv[2]);
  const int BlockSize  = std::stoi(argv[3]);  // GPU thread block size

  const int matrix_byte_size = M * N * sizeof(double);

  //Loading a synthetic tridiagonal matrix into our structure
  ThomasMatrix params = loadThomasMatrixSyn(M);

  // Allocate host arrays for CPU execution 
  double* u_seq = (double*) malloc(matrix_byte_size);
  double* u_Thomas_host =  (double*) malloc(matrix_byte_size);
  double* u_input = (double*) malloc(matrix_byte_size);

  double* d_seq = (double*) malloc(matrix_byte_size);
  double* d_Thomas_host =  (double*) malloc(matrix_byte_size);
  double* d_input = (double*) malloc(matrix_byte_size);

  double* l_seq = (double*) malloc(matrix_byte_size);
  double* l_Thomas_host =  (double*) malloc(matrix_byte_size);
  double* l_input = (double*) malloc(matrix_byte_size);

  double* rhs_seq = (double*) malloc(matrix_byte_size);
  double* rhs_Thomas_host = (double*) malloc(matrix_byte_size);
  double* rhs_input = (double*) malloc(matrix_byte_size);

  double* rhs_seq_output = (double*) malloc(matrix_byte_size);
  double* rhs_seq_interleave = (double*) malloc(matrix_byte_size);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      u_seq[(i * M) + j] = params.a[j];
      u_input[(i * M) + j] = params.a[j];

      d_seq[(i * M) + j] = params.d[j];
      d_input[(i * M) + j] = params.d[j];

      l_seq[(i * M) + j] = params.b[j];
      l_input[(i * M) + j] = params.b[j];

      rhs_seq[(i * M) + j] = params.rhs[j];
      rhs_input[(i * M) + j] = params.rhs[j];

    }
  }


  // Sequantial CPU Execution for correct error check
  for (int n = 0; n < 100; n++) { 
    solve_seq( l_seq, d_seq, u_seq, rhs_seq, M, N );
  }

  for (int i = 0; i < M*N; ++i) {
    rhs_seq_output[i] = rhs_seq[i];
  }

  // Initialize again because u_seq and rhs_seq are modified by solve_seq
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      u_seq[(i * M) + j] = params.a[j];
      u_input[(i * M) + j] = params.a[j];

      d_seq[(i * M) + j] = params.d[j];
      d_input[(i * M) + j] = params.d[j];

      l_seq[(i * M) + j] = params.b[j];
      l_input[(i * M) + j] = params.b[j];

      rhs_seq[(i * M) + j] = params.rhs[j];
      rhs_input[(i * M) + j] = params.rhs[j];
    }
  }


  // Transpose the inputs for sequential accesses on a GPU 
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      u_Thomas_host[i*N+j] = u_input[j*M+i];
      l_Thomas_host[i*N+j] = l_input[j*M+i];
      d_Thomas_host[i*N+j] = d_input[j*M+i];
      rhs_Thomas_host[i*N+j] = rhs_input[j*M+i];
      rhs_seq_interleave[i*N+j] = rhs_seq_output[j*M+i];
    }
  }

  { // sycl scope
 
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);


  buffer<double, 1> u_device (u_Thomas_host, N * M);
  buffer<double, 1> d_device (d_Thomas_host, N * M);
  buffer<double, 1> l_device (l_Thomas_host, N * M);
  buffer<double, 1> rhs_device (rhs_Thomas_host, N * M);
  d_device.set_final_data(nullptr);

  range<1> gws = (N + BlockSize - 1) / BlockSize * BlockSize; 
  range<1> lws = BlockSize;

  for (int n = 0; n < 100; n++) {

    q.submit([&] (handler &cgh) {
      auto L = l_device.get_access<sycl_read>(cgh);
      auto D = d_device.get_access<sycl_read>(cgh);
      auto U = u_device.get_access<sycl_read_write>(cgh);
      auto RHS = rhs_device.get_access<sycl_read_write>(cgh);

      cgh.parallel_for<class thomas>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
    
        int tid = item.get_global_id(0);
        if (tid < N) {
          int first = tid;
          int last  = N*(M-1)+tid;

          U[first] /= D[first];
          RHS[first] /= D[first];

          for (int i = first + N; i < last; i+=N) {
            U[i] /= D[i] - L[i] * U[i-N];
            RHS[i] = ( RHS[i] - L[i] * RHS[i-N] ) / 
                     ( D[i] - L[i] * U[i-N] );
          }

          RHS[last] = ( RHS[last] - L[last] * RHS[last-N] ) / 
                      ( D[last] - L[last] * U[last-N] );

          for (int i = last-N; i >= first; i-=N) {
            RHS[i] -= U[i] * RHS[i+N];
          }
        }
      });
    });
  }
  q.wait();

  }

  // verify
  calcError(rhs_seq_interleave,rhs_Thomas_host,N*M);


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
  free(rhs_seq_interleave);

  return 0;
}


