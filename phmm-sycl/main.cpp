#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include "common.h"
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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  fArray *d_cur_forward = malloc_device<fArray>(forward_matrix_elem, q);
  q.memcpy(d_cur_forward, h_cur_forward, forward_matrix_size);

  fArray *d_next_forward = malloc_device<fArray>(forward_matrix_elem, q);

  fArray *d_emis = malloc_device<fArray>(emissions_elem, q);
  q.memcpy(d_emis, h_emis, emissions_size);

  tArray *d_trans = malloc_device<tArray>(transitions_elem, q);
  q.memcpy(d_trans, h_trans, transitions_size);

  lArray *d_like = malloc_device<lArray>(likelihood_elem, q);
  q.memcpy(d_like, h_like, likelihood_size);

  sArray *d_start = malloc_device<sArray>(start_transitions_elem, q);
  q.memcpy(d_start, h_start, start_transitions_size);

  range<1> gws (batch * (states-1));
  range<1> lws (batch);

  q.wait();
  auto t1 = std::chrono::high_resolution_clock::now();

  for(int count = 0; count < repeat; count++) {
    for (int i = 1; i < x_dim + 1; i++) {
      for (int j = 1; j < y_dim + 1; j++) {
        q.submit([&] (handler &cgh) {
          cgh.parallel_for<class pair_hmm_forward>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            pair_HMM_forward(item, i, j, d_cur_forward, d_trans, d_emis,
                             d_like, d_start, d_next_forward);
          });
        });
        auto t = d_cur_forward;
        d_cur_forward = d_next_forward;
        d_next_forward = t;
      }
    }
  }
  q.wait();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> milli = (t2 - t1);
  std::cout << "Total execution time " <<  milli.count() << " milliseconds\n" ;

  q.memcpy(h_cur_forward, d_cur_forward, forward_matrix_size).wait();

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

  free(d_cur_forward, q);
  free(d_next_forward, q);
  free(d_emis, q);
  free(d_trans, q);
  free(d_like, q);
  free(d_start, q);
  free(h_cur_forward);
  free(h_emis);
  free(h_trans);
  free(h_like);
  free(h_start);

  return 0;
}
