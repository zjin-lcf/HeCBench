#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"
#include "utils.h"
#include "kernels.cpp"

int main(int argc, char *argv[])
{
  double t_start = 0;
  double dt = 0.02E-3;

  int num_timesteps = 1000000;
  int num_nodes = 1; 

  if (argc > 1) {
    num_timesteps = atoi(argv[1]);
    printf("num_timesteps set to %d\n", num_timesteps);

    num_nodes = atoi(argv[2]);
    printf("num_nodes set to %d\n", num_nodes);

    if(num_timesteps <= 0 || num_nodes <= 0)
      exit(EXIT_FAILURE);
  }

  unsigned int num_states = NUM_STATES;
  size_t total_num_states = num_nodes * num_states;
  size_t states_size = total_num_states * sizeof(double);
  double *states = (double*) malloc(states_size);
  init_state_values(states, num_nodes);

  double *states2 = (double*) malloc(states_size);
  memcpy(states2, states, states_size);

  unsigned int num_parameters = NUM_PARAMS;
  size_t total_num_parameters = num_nodes * num_parameters;
  size_t parameters_size = total_num_parameters * sizeof(double);
  double *parameters = (double*) malloc(parameters_size);
  init_parameters_values(parameters, num_nodes);

  double t = t_start;

  struct timespec timestamp_start, timestamp_now;
  double time_elapsed;

  printf("CPU: Rush Larsen (exp integrator on all gates)\n");
  for (int it = 0; it < num_timesteps; it++) {
    forward_rush_larsen(states, t, dt, parameters, num_nodes);
    t += dt;
  }

  printf("GPU: Rush Larsen (exp integrator on all gates)\n");

  {  // sycl scope
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<double, 1> d_states(states2, total_num_states);
  buffer<double, 1> d_parameters(parameters, total_num_parameters); 

  // All nodes run the same kernel
  clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_start);
  range<1> gws ((num_nodes + 255)/256 * 256);
  range<1> lws (256);

  t = t_start;
  for (int it = 0; it < num_timesteps; it++) {
    q.submit([&] (handler &cgh) {
      auto states_acc = d_states.get_access<sycl_read_write>(cgh);
      auto params_acc = d_parameters.get_access<sycl_read>(cgh);
      cgh.parallel_for<class k>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        k_forward_rush_larsen(item, states_acc.get_pointer(), t, dt, 
                              params_acc.get_pointer(), num_nodes); 
      });
    });
    t += dt;
  }
  q.wait();
  clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_now);
  time_elapsed = timestamp_now.tv_sec - timestamp_start.tv_sec + 1E-9 * (timestamp_now.tv_nsec - timestamp_start.tv_nsec);
  printf("Computed %d time steps in %g s. Time steps per second: %g\n",
      num_timesteps, time_elapsed, num_timesteps/time_elapsed);
  printf("\n");

  }

  double rmse = 0.0;
  for (size_t i = 0; i < total_num_states; i++) {
    rmse += (states2[i] - states[i]) * (states2[i] - states[i]);
#ifdef VERBOSE
    printf("state[%d] = %lf\n", i, states[i]);
#endif
  }
  printf("RMSE = %lf\n", sqrt(rmse / (total_num_states)));
 
  free(states);
  free(states2);
  free(parameters);

  return 0;
}
