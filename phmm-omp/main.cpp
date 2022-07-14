#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include <omp.h>
#include "constants_types.h"
#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  size_t forward_matrix_elem = (x_dim+1)*(y_dim+1)*batch*(states-1);
  size_t emissions_elem = (x_dim+1)*(y_dim+1)*batch*(states-1);
  size_t transitions_elem = (x_dim+1)*(states-1)*states*batch;
  size_t start_transitions_elem = batch*(states-1);
  size_t likelihood_elem = 2*2*(states-1)*batch;

  size_t forward_matrix_size = forward_matrix_elem * sizeof(double);
  size_t emissions_size = emissions_elem * sizeof(double);
  size_t transitions_size = transitions_elem * sizeof(double);
  size_t start_transitions_size = start_transitions_elem * sizeof(double);
  size_t likelihood_size = likelihood_elem * sizeof(double);

  fArray *h_cur_forward = (fArray*) malloc (forward_matrix_size); 
  fArray *h_emis = (fArray*) malloc (emissions_size);
  tArray *h_trans = (tArray*) malloc (transitions_size);
  lArray *h_like = (lArray*) malloc (likelihood_size);
  sArray *h_start = (sArray*) malloc (start_transitions_size);

  std::default_random_engine rng (123);
  std::uniform_real_distribution<double> dist (0.0, 1.0);
  for (int i = 0; i < x_dim+1; i++) {
    for (int j = 0; j < y_dim+1; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states-1; s++) {
          h_cur_forward[i][j][b][s] = dist(rng);
          h_emis[i][j][b][s] = dist(rng);
        }
      }
    }
  }

  for (int i = 0; i < x_dim+1; i++) {
    for (int b = 0; b < batch; b++) {
      for (int s = 0; s < states-1; s++) {
        for (int t = 0; t < states; t++) {
          h_trans[i][b][s][t] = dist(rng);
        }
      }
    }
  }
         
  for (int i = 0; i < batch; i++) {
    for (int s = 0; s < states-1; s++) {
      h_start[i][s] = dist(rng);
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j< 2; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states-1; s++) {
          h_like[i][j][b][s] = dist(rng);
        }
      }
    }
  }

  double *d_cur_forward = (double*) h_cur_forward;
  double *d_emis = (double*) h_emis;
  double *d_trans = (double*) h_trans;
  double *d_like = (double*) h_like;
  double *d_start = (double*) h_start;
  double *d_next_forward = (double*) malloc (forward_matrix_size);

  #pragma omp target data map (tofrom: d_cur_forward[0:forward_matrix_elem]),\
                          map (to: d_emis[0:forward_matrix_elem],\
                                   d_trans[0:transitions_elem],\
                                   d_like[0:likelihood_elem],\
                                   d_start[0:start_transitions_elem]) \
                          map (alloc: d_next_forward[0:forward_matrix_elem])
  {
    // OMP teams and threads
    const int num_teams = batch;
    const int num_threads = states-1;

    auto t1 = std::chrono::high_resolution_clock::now();

    for(int count = 0; count < repeat; count++) {
      for (int i = 1; i < x_dim + 1; i++) {
        for (int j = 1; j < y_dim + 1; j++) {
          pair_HMM_forward(num_teams, num_threads, i, j, 
                           (fArray*)d_cur_forward, (tArray*)d_trans,
                           (fArray*)d_emis, (lArray*)d_like,
                           (sArray*)d_start, (fArray*)d_next_forward);
          auto t = d_cur_forward;
          d_cur_forward = d_next_forward;
          d_next_forward = t;
        }
      }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> milli = (t2 - t1);
    std::cout << "Total execution time " <<  milli.count() << " milliseconds\n" ;
  }

  double checkSum = 0.0;
  for (int i = 0; i < x_dim+1; i++) {
    for (int j = 0; j < y_dim+1; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states-1; s++) {
          #ifdef DEBUG
          std::cout << h_cur_forward[i][j][b][s] << std::endl;
          #endif
          checkSum += h_cur_forward[i][j][b][s];
        }
      }
    }
  }
  std::cout << "Checksum " << checkSum << std::endl;

  free(h_cur_forward);
  free(h_emis);
  free(h_trans);
  free(h_like);
  free(h_start);
  free(d_next_forward);

  return 0;
}
