#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include "common.h"
#include "constants_types.h"

int main() {

  range<1> gws (batch * (states-1));
  range<1> lws (batch);

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
  queue q(dev_sel);

  buffer<double, 1> d_cur_forward ((double*)h_cur_forward, forward_matrix_elem);
  buffer<double, 1> d_next_forward (forward_matrix_elem);
  buffer<double, 1> d_emis ((double*)h_emis, emissions_elem);
  buffer<double, 1> d_trans ((double*)h_trans, transitions_elem);
  buffer<double, 1> d_like ((double*)h_like, likelihood_elem);
  buffer<double, 1> d_start ((double*)h_start, start_transitions_elem);

  auto d_cur_fw_re = d_cur_forward.reinterpret<fArray>(range<1>(x_dim+1));
  auto d_nxt_fw_re = d_next_forward.reinterpret<fArray>(range<1>(x_dim+1));
  auto d_emis_re = d_emis.reinterpret<fArray>(range<1>(x_dim+1));
  auto d_trans_re = d_trans.reinterpret<tArray>(range<1>(x_dim+1));
  auto d_like_re = d_like.reinterpret<lArray>(range<1>(2));
  auto d_start_re = d_start.reinterpret<sArray>(range<1>(batch));

  q.wait();
  auto t1 = std::chrono::high_resolution_clock::now();

  for(int count = 0; count < 100; count++) {
    for (int cur_i = 1; cur_i < x_dim + 1; cur_i++) {
      for (int cur_j = 1; cur_j < y_dim + 1; cur_j++) {
        q.submit([&] (handler &cgh) {
          auto forward_matrix_in = d_cur_fw_re.get_access<sycl_read>(cgh);
          auto forward_matrix_out = d_nxt_fw_re.get_access<sycl_write>(cgh);
          auto emissions = d_emis_re.get_access<sycl_read>(cgh);
          auto transitions = d_trans_re.get_access<sycl_read>(cgh);
          auto likelihood = d_like_re.get_access<sycl_read>(cgh);
          auto start_transitions = d_start_re.get_access<sycl_read>(cgh);
          accessor<double, 2, sycl_read_write, access::target::local> e({batch, states-1}, cgh);
          accessor<double, 3, sycl_read_write, access::target::local> f01({1, batch, 2}, cgh);
          accessor<double, 3, sycl_read_write, access::target::local> mul_3d({1, batch, 2}, cgh);
          accessor<double, 4, sycl_read_write, access::target::local> mul_4d({4, batch, 1, 2}, cgh);
          cgh.parallel_for<class pair_hmm_forward>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            #include "kernel.h"
          });
        });
        auto t = std::move(d_cur_fw_re);
        d_cur_fw_re = std::move(d_nxt_fw_re);
        d_nxt_fw_re = std::move(t);
      }
    }
  }
  q.wait();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> milli = (t2 - t1);
  std::cout << "Total execution time " <<  milli.count() << " milliseconds\n" ;

  q.submit([&] (handler &cgh) {
    auto acc = d_cur_forward.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_cur_forward);
  }).wait();

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

  return 0;
}
