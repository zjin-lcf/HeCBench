#include <chrono>
#include <random>
#include <new>
#include "common.h"
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

  { // sycl scope

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_a (a, ncells);
  buffer<float, 1> d_b (b, ncells);
  buffer<float, 1> d_dx2 (ncells);
  buffer<float, 1> d_dy2 (ncells);
  buffer<float, 1> d_dz2 (ncells);
  buffer<float, 1> d_ra (ncells);
  buffer<float, 1> d_rb (ncells);
  buffer<float, 1> d_da (ncells);
  buffer<float, 1> d_db (ncells);

  // copy data to device
  q.submit([&] (handler &cgh) {
    auto acc = d_dx2.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_dy2.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_dz2.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_ra.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_rb.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_da.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_db.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  // set constants
  float diffcon_a = Da / (dx * dx);
  float diffcon_b = Db / (dx * dx);

  range<2> gws_x (mz*pencils, mx*my/pencils);
  range<2> lws_x (pencils, mx);

  range<2> gws_y (mz*my, mx);
  range<2> lws_y (my, pencils);

  range<2> gws_z (mz*my, mx);
  range<2> lws_z (mz, pencils);

  range<1> gws ((ncells + mx - 1) / mx * mx);
  range<1> lws (mx);

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
      q.submit([&] (handler &cgh) {
        auto a = d_a.get_access<sycl_read>(cgh); 
        auto dx2 = d_dx2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class x2_zeroflux_a>(nd_range<2>(gws_x, lws_x), [=] (nd_item<2> item) {
          derivative_x2_zeroflux(a.get_pointer(), dx2.get_pointer(),
                                 sf.get_pointer(), item, mx, my);
        });
      });

      // y2 derivative
      q.submit([&] (handler &cgh) {
        auto a = d_a.get_access<sycl_read>(cgh); 
        auto dy2 = d_dy2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class y2_zeroflux_a>(nd_range<2>(gws_y, lws_y), [=] (nd_item<2> item) {
          derivative_y2_zeroflux(a.get_pointer(), dy2.get_pointer(),
                                 sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (handler &cgh) {
        auto a = d_a.get_access<sycl_read>(cgh); 
        auto dz2 = d_dz2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class z2_zeroflux_a>(nd_range<2>(gws_z, lws_z), [=] (nd_item<2> item) {
          derivative_z2_zeroflux(a.get_pointer(), dz2.get_pointer(),
                                 sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    } else {
      // x2 derivative
      q.submit([&] (handler &cgh) {
        auto a = d_a.get_access<sycl_read>(cgh); 
        auto dx2 = d_dx2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class x2_pbc_a>(nd_range<2>(gws_x, lws_x), [=] (nd_item<2> item) {
          derivative_x2_pbc(a.get_pointer(), dx2.get_pointer(),
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // y2 derivative
      q.submit([&] (handler &cgh) {
        auto a = d_a.get_access<sycl_read>(cgh); 
        auto dy2 = d_dy2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class y2_pb_a>(nd_range<2>(gws_y, lws_y), [=] (nd_item<2> item) {
          derivative_y2_pbc(a.get_pointer(), dy2.get_pointer(),
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (handler &cgh) {
        auto a = d_a.get_access<sycl_read>(cgh); 
        auto dz2 = d_dz2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class z2_pbc_a>(nd_range<2>(gws_z, lws_z), [=] (nd_item<2> item) {
          derivative_z2_pbc(a.get_pointer(), dz2.get_pointer(),
                            sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    }

    // sum all three derivative components
    q.submit([&] (handler &cgh) {
      auto da = d_da.get_access<sycl_discard_write>(cgh); 
      auto dx2 = d_dx2.get_access<sycl_read>(cgh); 
      auto dy2 = d_dy2.get_access<sycl_read>(cgh); 
      auto dz2 = d_dz2.get_access<sycl_read>(cgh); 
      cgh.parallel_for<class sum_a>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        construct_laplacian(da.get_pointer(), dx2.get_pointer(), dy2.get_pointer(), 
                            dz2.get_pointer(), item, ncells, diffcon_a);
      });
    });

    // calculate laplacian for B
    if(zeroflux) {
      // x2 derivative
      q.submit([&] (handler &cgh) {
        auto b = d_b.get_access<sycl_read>(cgh); 
        auto dx2 = d_dx2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class x2_zeroflux_b>(nd_range<2>(gws_x, lws_x), [=] (nd_item<2> item) {
          derivative_x2_zeroflux(b.get_pointer(), dx2.get_pointer(), 
                                 sf.get_pointer(), item, mx, my);
        });
      });

      // y2 derivative
      q.submit([&] (handler &cgh) {
        auto b = d_b.get_access<sycl_read>(cgh); 
        auto dy2 = d_dy2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class y2_zeroflux_b>(nd_range<2>(gws_y, lws_y), [=] (nd_item<2> item) {
          derivative_y2_zeroflux(b.get_pointer(), dy2.get_pointer(),
                                 sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (handler &cgh) {
        auto b = d_b.get_access<sycl_read>(cgh); 
        auto dz2 = d_dz2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class z2_zeroflux_b>(nd_range<2>(gws_z, lws_z), [=] (nd_item<2> item) {
          derivative_z2_zeroflux(b.get_pointer(), dz2.get_pointer(),
                                 sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    } else {
      // x2 derivative
      q.submit([&] (handler &cgh) {
        auto b = d_b.get_access<sycl_read>(cgh); 
        auto dx2 = d_dx2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class x2_pbc_b>(nd_range<2>(gws_x, lws_x), [=] (nd_item<2> item) {
          derivative_x2_pbc(b.get_pointer(), dx2.get_pointer(),
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // y2 derivative
      q.submit([&] (handler &cgh) {
        auto b = d_b.get_access<sycl_read>(cgh); 
        auto dy2 = d_dy2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class y2_pb_b>(nd_range<2>(gws_y, lws_y), [=] (nd_item<2> item) {
          derivative_y2_pbc(b.get_pointer(), dy2.get_pointer(), 
                            sf.get_pointer(), item, mx, my, pencils);
        });
      });

      // z2 derivative
      q.submit([&] (handler &cgh) {
        auto b = d_b.get_access<sycl_read>(cgh); 
        auto dz2 = d_dz2.get_access<sycl_discard_write>(cgh); 
        accessor<float, 1, sycl_read_write, access::target::local> sf (shared_mem_size, cgh);
        cgh.parallel_for<class z2_pbc_b>(nd_range<2>(gws_z, lws_z), [=] (nd_item<2> item) {
          derivative_z2_pbc(b.get_pointer(), dz2.get_pointer(),
                            sf.get_pointer(), item, mx, my, mz, pencils);
        });
      });
    }

    // sum all three derivative components
    q.submit([&] (handler &cgh) {
      auto db = d_db.get_access<sycl_discard_write>(cgh); 
      auto dx2 = d_dx2.get_access<sycl_read>(cgh); 
      auto dy2 = d_dy2.get_access<sycl_read>(cgh); 
      auto dz2 = d_dz2.get_access<sycl_read>(cgh); 
      cgh.parallel_for<class sum_b>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        construct_laplacian(db.get_pointer(), dx2.get_pointer(), dy2.get_pointer(), 
                            dz2.get_pointer(), item, ncells, diffcon_b);
      });
    });

    // calculate reaction
    q.submit([&] (handler &cgh) {
      auto a = d_a.get_access<sycl_read>(cgh); 
      auto b = d_b.get_access<sycl_read>(cgh); 
      auto ra = d_ra.get_access<sycl_discard_write>(cgh); 
      auto rb = d_rb.get_access<sycl_discard_write>(cgh); 
      cgh.parallel_for<class gray_scott>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        reaction_gray_scott(a.get_pointer(), b.get_pointer(), 
                            ra.get_pointer(), rb.get_pointer(), item, ncells, c1, c2);
      });
    });

    // update
    q.submit([&] (handler &cgh) {
      auto a = d_a.get_access<sycl_read_write>(cgh);
      auto b = d_b.get_access<sycl_read_write>(cgh);
      auto da = d_da.get_access<sycl_read>(cgh);
      auto db = d_db.get_access<sycl_read>(cgh);
      auto ra = d_ra.get_access<sycl_read>(cgh);
      auto rb = d_rb.get_access<sycl_read>(cgh);
      cgh.parallel_for<class integrate>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        update(a.get_pointer(), b.get_pointer(), da.get_pointer(), db.get_pointer(), 
               ra.get_pointer(), rb.get_pointer(), item, ncells, dt);
      });
    });
  }

  q.wait();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("timesteps: %d\n", timesteps);
  printf("Total kernel execution time:     %12.3f s\n\n", elapsed_seconds.count());

  } // sycl scope

  // output lowest and highest values
  stats(a, b, ncells);

  delete [] a;
  delete [] b;
  return 0;
}
