#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include "kernels.h"

// Copies the flow velocity from GPU to CPU memory, for data output.
void u_read(sycl::queue &q, lbm_vars *h_vars, lbm_vars *d_vars, int nl) {
  q.memcpy(h_vars->u_star.u0, d_vars->u_star.u0, sizeof(u_type) * nl);
  q.memcpy(h_vars->u_star.u1, d_vars->u_star.u1, sizeof(u_type) * nl);
  q.memcpy(h_vars->u_star.u2, d_vars->u_star.u2, sizeof(u_type) * nl);
  q.wait();
}

void lbm_u_alloc(lbm_u* u, size_t nl) {
  u->u0 = (u_type*)malloc(sizeof(u_type)*nl*3);
  u->u1 = u->u0 + nl;
  u->u2 = u->u1 + nl;
}

void lbm_u_gpu_alloc(sycl::queue &q, lbm_u *u, size_t nl) {
  u->u0 = (u_type *)sycl::malloc_device(sizeof(u_type) * nl * 3, q);
  q.memset(u->u0, 0, sizeof(u_type) * nl * 3);
  u->u1 = u->u0 + nl;
  u->u2 = u->u1 + nl;
}

void lbm_vars_alloc(lbm_vars* vars, lattice_type nb_dir, int nl, int vnl){

  lbm_u_alloc(&vars->u, vnl);
  lbm_u_alloc(&vars->u_star, vnl);
  lbm_u_alloc(&vars->g, vnl);
  vars->f0 = (double*)malloc(nl*nb_dir*sizeof(double));
  vars->f1 = (double*)malloc(nl*nb_dir*sizeof(double));
}

void lbm_vars_gpu_alloc(sycl::queue &q, lbm_vars *vars, lattice_type nb_dir, int nl, int vnl) {
  lbm_u_gpu_alloc(q, &vars->u_star, vnl);
  lbm_u_gpu_alloc(q, &vars->g, vnl);

  vars->f0 = sycl::malloc_device<double>(nl * nb_dir, q);
  vars->f1 = sycl::malloc_device<double>(nl * nb_dir, q);
  vars->boundary_flag = sycl::malloc_device<flag_type>(nl, q);
  vars->boundary_values = sycl::malloc_device<int>(nl, q);
  vars->boundary_dirs = sycl::malloc_device<flag_type>(nl, q);
  vars->r = sycl::malloc_device<double>(vnl, q);
}

void init_and_allocate_data(sycl::queue &q, BoxCU &domain, lbm_vars *h_vars, lbm_vars *d_vars) {

  const int nl = domain.nx*domain.ny*domain.nz;
  const int vnl = nl;

  // C_dirs and C_p are located in constant memory in the CUDA program
  char* C_dirs = sycl::malloc_device<char>(81, q);
  q.memcpy(C_dirs, dirs, 57 * sizeof(char));

  char* C_p = sycl::malloc_device<char>(12, q);
  q.memcpy(C_p, outer_bounds_priority, sizeof(outer_bounds_priority));

  // Hold all pointers needed for the lbm computation. There is one for the GPU and one for the CPU

  lbm_vars_alloc(h_vars, D3Q19, nl, vnl);
  lbm_vars_gpu_alloc(q, d_vars, D3Q19, nl, vnl);

  outer_wall wall{type_b::bounce, type_b::bounce, type_b::bounce, type_b::bounce, type_b::moving_wall, type_b::bounce};

  // Initialization of flags according to wall
  q.memset(d_vars->boundary_flag, type_b::fluid, nl * sizeof(flag_type));

  q.submit([&](sycl::handler &cgh) {

    auto d_vars_boundary_flag = d_vars->boundary_flag;
    auto d_vars_boundary_values = d_vars->boundary_values;
    auto d_vars_boundary_dirs = d_vars->boundary_dirs;

    cgh.parallel_for<class init_flags>(
        sycl::nd_range<3>(sycl::range<3>(domain.nz, domain.ny, domain.nx),
                          sycl::range<3>(1, 1, domain.nx)),
      [=](sycl::nd_item<3> item) {
      make_flag(d_vars_boundary_flag, d_vars_boundary_values,
                d_vars_boundary_dirs, domain, wall, domain.nx,
                domain.ny, domain.nz, 0, item, C_p);
    });
  });

  q.submit([&](sycl::handler &cgh) {

    auto d_vars_boundary_flag = d_vars->boundary_flag;
    auto d_vars_boundary_dirs = d_vars->boundary_dirs;
    auto d_vars_boundary_values = d_vars->boundary_values;

    cgh.parallel_for<class init_vals>(
        sycl::nd_range<3>(sycl::range<3>(domain.nz, domain.ny, domain.nx),
                          sycl::range<3>(1, 1, domain.nx)),
      [=](sycl::nd_item<3> item) {
      find_wall<D3Q19>(d_vars_boundary_flag, d_vars_boundary_dirs,
                       d_vars_boundary_values, domain, 0, item, C_dirs);
    });
  });

  q.wait();
  sycl::free(C_dirs, q);
  sycl::free(C_p, q);
}

void deallocate_data(sycl::queue &q, lbm_vars *h_vars, lbm_vars *d_vars) {
  free(h_vars->u.u0);
  free(h_vars->u_star.u0);
  free(h_vars->g.u0);
  free(h_vars->f0);
  free(h_vars->f1);
  sycl::free(d_vars->u_star.u0, q);
  sycl::free(d_vars->g.u0, q);
  sycl::free(d_vars->f0, q);
  sycl::free(d_vars->f1, q);
  sycl::free(d_vars->boundary_flag, q);
  sycl::free(d_vars->boundary_values, q);
  sycl::free(d_vars->boundary_dirs, q);
  sycl::free(d_vars->r, q);
}

double run_benchmark(sycl::queue &q, BoxCU &domain, lbm_vars h_vars, lbm_vars d_vars) {
  int N = domain.ny;

  const int bench_ini_iter = 1000;
  const int bench_max_iter = 2000;
  const int output_frame = 2000;

  const double ulb = 0.02;
  const double dx = 1. / (N - 2.);
  const double dt = dx * ulb;

  const double Re = 100.;
  const double nu = ulb * (N - 2.) / Re;
  const double omega = 1. / (3. * nu + 0.5);

  printf("omega = %f\n", omega);

  const int nl = domain.nx*domain.ny*domain.nz;

  // Initialization of the populations.
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class init_population>(
        sycl::nd_range<3>(sycl::range<3>(domain.nz, domain.ny, domain.nx),
                          sycl::range<3>(1, 1, domain.nx)),
      [=](sycl::nd_item<3> item) {
        init_velocity_g<D3Q19>(d_vars, domain, domain, domain.nz, 0, 0, 0, 1., item);
    });
  });

  int iter = 0;
  int num_bench_iter = 0;

  clock_t start = clock();
  clock_t end = 0;

  printf("Starting %d warmup iterations\n", bench_ini_iter);
  // Main time loop of the simulation.
  for(iter = 0; iter < bench_max_iter; ++iter) {

    bool do_output = iter < bench_ini_iter && iter > 0 && (iter % output_frame == 0 || iter == 149);
    if (iter == bench_ini_iter) {
      printf("Starting %d benchmark iterations\n", bench_max_iter - bench_ini_iter);
      start = clock();
    }
    if (iter >= bench_ini_iter) {
      ++num_bench_iter;
    }

    // LBM collision-streaming cycle, in parallel over every cell. Stream precedes collision.
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class collision_streaming>(
        sycl::nd_range<3>(
            sycl::range<3>(domain.nz, domain.ny, ((domain.nx - 1) / 64 + 1) * 64),
            sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item) {
          collide_and_stream_g<D3Q19>(d_vars, domain, ulb, omega, do_output, iter, item);
      });
    });

    // Swap populations pointer, f0 are population to be read and f1 the population to be written,
    // this is the double population soa scheme.
    double *tp = d_vars.f0;
    d_vars.f0 = d_vars.f1;
    d_vars.f1 = tp;
    q.wait();

    // Ouput average kinetic energy for validation.
    if (do_output) {
      u_read(q, &h_vars, &d_vars, nl);

      double energy = 0;

      for(int z = 0; z < domain.nz; ++z){
        for(int y = 0; y < domain.ny; ++y){
          for(int x = 0; x < domain.nx; ++x){
            energy += h_vars.u_star.u0[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] *
              h_vars.u_star.u0[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] +
              h_vars.u_star.u1[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] *
              h_vars.u_star.u1[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] +
              h_vars.u_star.u2[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] *
              h_vars.u_star.u2[IDX(x,y,z, domain.nx, domain.ny, domain.nz)];
          }
        }
      }
      energy *= 0.5;

      printf("energy %f iteration %d \n", energy*dx*dx/(dt*dt), iter);

      if (iter == 149 && N == 102) {
        printf("Regression test at iteration %d: Average energy LU = %f", iter, energy);
        const double reference_energy = 2.09868507623;
        if (fabs(energy - reference_energy) < 1.e-7) {
          printf(": OK\n");
        }
        else {
          printf(": FAILED\nExpected the value %f\n", reference_energy);
        }
      }
    }
  }
  end = clock();
  double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
  double mlups = ((double)nl*(double)num_bench_iter / elapsed) / 1.e6;
  return mlups;
}

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: %s <N>\n", argv[0]);
    printf("N: domain size in each dimension\n");
    return -1;
  }
  const int N = atoi(argv[1]);

  BoxCU domain;
  domain.nx = N;
  domain.ny = N;
  domain.nz = N;
  lbm_vars h_vars, d_vars;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  init_and_allocate_data(q, domain, &h_vars, &d_vars);

  double mlups[10];
  for (int i = 0; i < 10; ++i)
    mlups[i] = run_benchmark(q, domain, h_vars, d_vars);

  deallocate_data(q, &h_vars, &d_vars);

  printf("\nPerformance: MLUPS\n");
  for (int i = 0; i < 10; ++i) {
    printf("%.4f\n", mlups[i]);
  }

  return 0;
}
