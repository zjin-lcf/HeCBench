#include <chrono>
#include <random>
#include <new>
#include <sycl/sycl.hpp>
#include "util.h"
#include "kernels.cpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <timesteps>\n", argv[0]);
    return 1;
  }
  unsigned int timesteps = atoi(argv[1]);

  unsigned int mx = 128;
  unsigned int my = 128;
  unsigned int mz = 128;
  unsigned int ncells = mx * my * mz;
  unsigned int pencils = 2;
  bool zeroflux = true;

  // reaction settings of kinetic system
  float Da = 0.16;            // diffusion constant of A
  float Db = 0.08;            // diffusion constant of B
  float dt = 0.25;            // temporal discretization
  float dx = 0.5;             // spatial discretization

  // generalized kinetic parameters
  float c1 = 0.0392;
  float c2 = 0.0649;

  printf("Starting time-integration\n");
  // build initial concentrations
  printf("Constructing initial concentrations...\n");
  // concentration of components A and B
  float* a = new float[ncells];
  float* b = new float[ncells];

  build_input_central_cube(ncells, mx, my, mz, a, b, 1.0f, 0.0f, 0.5f, 0.25f, 0.05f);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_a = sycl::malloc_device<float>(ncells, q);
  float *d_b = sycl::malloc_device<float>(ncells, q);
  float *d_dx2 = sycl::malloc_device<float>(ncells, q);
  float *d_dy2 = sycl::malloc_device<float>(ncells, q);
  float *d_dz2 = sycl::malloc_device<float>(ncells, q);
  float *d_ra = sycl::malloc_device<float>(ncells, q);
  float *d_rb = sycl::malloc_device<float>(ncells, q);
  float *d_da = sycl::malloc_device<float>(ncells, q);
  float *d_db = sycl::malloc_device<float>(ncells, q);

  // copy data to device
  const size_t bytes = ncells * sizeof(float);
  q.memcpy(d_a, a, bytes);
  q.memcpy(d_b, b, bytes);
  q.memset(d_dx2, 0, bytes);
  q.memset(d_dy2, 0, bytes);
  q.memset(d_dz2, 0, bytes);
  q.memset(d_ra, 0, bytes);
  q.memset(d_rb, 0, bytes);
  q.memset(d_da, 0, bytes);
  q.memset(d_db, 0, bytes);

  // set constants
  float diffcon_a = Da / (dx * dx);
  float diffcon_b = Db / (dx * dx);

  sycl::range<2> gws_x (mz*pencils, mx*my/pencils);
  sycl::range<2> lws_x (pencils, mx);

  sycl::range<2> gws_y (mz*my, mx);
  sycl::range<2> lws_y (my, pencils);

  sycl::range<2> gws_z (mz*my, mx);
  sycl::range<2> lws_z (mz, pencils);

  sycl::range<1> gws ((ncells + mx - 1) / mx * mx);
  sycl::range<1> lws (mx);

  unsigned shared_mem_size;
  if(zeroflux) {
    shared_mem_size = pencils * mx;
  } else {
    shared_mem_size = pencils * (mx + 2);
  }

  // keep track of time
  q.wait();
  auto start = std::chrono::system_clock::now();

  for(unsigned int t=0; t<timesteps; t++) {

    // calculate laplacian for A
    if(zeroflux) {
      // x2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class x2_zeroflux_a>(
          sycl::nd_range<2>(gws_x, lws_x), [=] (sycl::nd_item<2> item) {
          derivative_x2_zeroflux(d_a, d_dx2, sf.get_pointer(), item, mx, my);
        });
      });

      // y2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class y2_zeroflux_a>(
          sycl::nd_range<2>(gws_y, lws_y), [=] (sycl::nd_item<2> item) {
          derivative_y2_zeroflux(d_a, d_dy2,
                                 sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class z2_zeroflux_a>(
          sycl::nd_range<2>(gws_z, lws_z), [=] (sycl::nd_item<2> item) {
          derivative_z2_zeroflux(d_a, d_dz2,
                                 sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    } else {
      // x2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class x2_pbc_a>(
          sycl::nd_range<2>(gws_x, lws_x), [=] (sycl::nd_item<2> item) {
          derivative_x2_pbc(d_a, d_dx2,
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // y2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class y2_pb_a>(
          sycl::nd_range<2>(gws_y, lws_y), [=] (sycl::nd_item<2> item) {
          derivative_y2_pbc(d_a, d_dy2,
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class z2_pbc_a>(
          sycl::nd_range<2>(gws_z, lws_z), [=] (sycl::nd_item<2> item) {
          derivative_z2_pbc(d_a, d_dz2,
                            sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    }

    // sum all three derivative components
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class sum_a>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        construct_laplacian(d_da, d_dx2, d_dy2,
                            d_dz2, item, ncells, diffcon_a);
      });
    });

    // calculate laplacian for B
    if(zeroflux) {
      // x2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class x2_zeroflux_b>(
          sycl::nd_range<2>(gws_x, lws_x), [=] (sycl::nd_item<2> item) {
          derivative_x2_zeroflux(d_b, d_dx2,
                                 sf.get_pointer(), item, mx, my);
        });
      });

      // y2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class y2_zeroflux_b>(
          sycl::nd_range<2>(gws_y, lws_y), [=] (sycl::nd_item<2> item) {
          derivative_y2_zeroflux(d_b, d_dy2,
                                 sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class z2_zeroflux_b>(
          sycl::nd_range<2>(gws_z, lws_z), [=] (sycl::nd_item<2> item) {
          derivative_z2_zeroflux(d_b, d_dz2,
                                 sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    } else {
      // x2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class x2_pbc_b>(
          sycl::nd_range<2>(gws_x, lws_x), [=] (sycl::nd_item<2> item) {
          derivative_x2_pbc(d_b, d_dx2,
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // y2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class y2_pb_b>(
          sycl::nd_range<2>(gws_y, lws_y), [=] (sycl::nd_item<2> item) {
          derivative_y2_pbc(d_b, d_dy2,
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sf (sycl::range<1>(shared_mem_size), cgh);
        cgh.parallel_for<class z2_pbc_b>(
          sycl::nd_range<2>(gws_z, lws_z), [=] (sycl::nd_item<2> item) {
          derivative_z2_pbc(d_b, d_dz2,
                            sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    }

    // sum all three derivative components
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class sum_b>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        construct_laplacian(d_db, d_dx2, d_dy2,
                            d_dz2, item, ncells, diffcon_b);
      });
    });

    // calculate reaction
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class gray_scott>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        reaction_gray_scott(d_a, d_b, d_ra, d_rb,
                            item, ncells, c1, c2);
      });
    });

    // update
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class integrate>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        update(d_a, d_b, d_da, d_db,
               d_ra, d_rb, item, ncells, dt);
      });
    });
  }

  q.wait();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("timesteps: %d\n", timesteps);
  printf("Total kernel execution time:     %12.3f s\n\n", elapsed_seconds.count());

  // copy results back
  q.memcpy(a, d_a, bytes);
  q.memcpy(b, d_b, bytes);
  q.wait();

  // output lowest and highest values
  stats(a, b, ncells);

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_ra, q);
  sycl::free(d_rb, q);
  sycl::free(d_da, q);
  sycl::free(d_db, q);
  sycl::free(d_dx2, q);
  sycl::free(d_dy2, q);
  sycl::free(d_dz2, q);

  delete [] a;
  delete [] b;
  return 0;
}
