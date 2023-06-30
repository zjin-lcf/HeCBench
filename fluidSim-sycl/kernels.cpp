#include <stdio.h>
#include <chrono>
#include <sycl/sycl.hpp>

// Thread block size
#define GROUP_SIZE 256

// Calculates equilibrium distribution 
double ced(double rho, double weight, const sycl::double2 dir, const sycl::double2 u)
{
  double u2 = (u.x() * u.x()) + (u.y() * u.y());
  double eu = (dir.x() * u.x()) + (dir.y() * u.y());
  return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

// convert_sycl::int8() may be language specific
inline sycl::int8 newPos (const int p, const sycl::double8 &dir) {
  sycl::int8 np;
  np.s0() = p + (int)dir.s0();
  np.s1() = p + (int)dir.s1();
  np.s2() = p + (int)dir.s2();
  np.s3() = p + (int)dir.s3();
  np.s4() = p + (int)dir.s4();
  np.s5() = p + (int)dir.s5();
  np.s6() = p + (int)dir.s6();
  np.s7() = p + (int)dir.s7();
  return np;
}

void lbm (
    sycl::nd_item<2> &item, 
    const double *__restrict if0,
          double *__restrict of0, 
    const sycl::double4 *__restrict if1234,
          sycl::double4 *__restrict of1234,
    const sycl::double4 *__restrict if5678,
          sycl::double4 *__restrict of5678,
    const bool *__restrict type,
    const sycl::double8 dirX,
    const sycl::double8 dirY,
    const double *__restrict weight,
    double omega)
{
  uint idx = item.get_global_id(1);
  uint idy = item.get_global_id(0);
  uint width = item.get_global_range(1);
  uint height = item.get_global_range(0);
  uint pos = idx + width * idy;

  // Read input distributions
  double f0 = if0[pos];
  sycl::double4 f1234 = if1234[pos];
  sycl::double4 f5678 = if5678[pos];

  // intermediate results
  double e0;
  sycl::double4 e1234;
  sycl::double4 e5678;

  double rho; // Density
  sycl::double2 u;  // Velocity

  // Collide
  if(type[pos]) // Boundary
  {
    e0 = f0;
    // Swap directions 
    // f1234.xyzw = f1234.zwxy;
    e1234.x() = f1234.z();
    e1234.y() = f1234.w();
    e1234.z() = f1234.x();
    e1234.w() = f1234.y();

    // f5678.xyzw = f5678.zwxy;
    e5678.x() = f5678.z();
    e5678.y() = f5678.w();
    e5678.z() = f5678.x();
    e5678.w() = f5678.y();

    rho = 0;
    u = (sycl::double2)(0);
  }
  else // Fluid
  {
    // Compute Rho (density) of a cell by a reduction on f
    sycl::double4 temp = f1234 + f5678;
    temp.lo() += temp.hi();
    rho = f0 + temp.x() + temp.y();

    // Compute u (velocity) of a cell in x and y directions
    sycl::double4 x1234 = dirX.lo();
    sycl::double4 x5678 = dirX.hi();
    sycl::double4 y1234 = dirY.lo();
    sycl::double4 y5678 = dirY.hi();
    u.x() = (sycl::dot(f1234, x1234) + sycl::dot(f5678, x5678)) / rho;
    u.y() = (sycl::dot(f1234, y1234) + sycl::dot(f5678, y5678)) / rho;

    // Compute f with respect to space and time
    e0        = ced(rho, weight[0], {0, 0}, u);
    e1234.x() = ced(rho, weight[1], {dirX.s0(), dirY.s0()}, u);
    e1234.y() = ced(rho, weight[2], {dirX.s1(), dirY.s1()}, u);
    e1234.z() = ced(rho, weight[3], {dirX.s2(), dirY.s2()}, u);
    e1234.w() = ced(rho, weight[4], {dirX.s3(), dirY.s3()}, u);
    e5678.x() = ced(rho, weight[5], {dirX.s4(), dirY.s4()}, u);
    e5678.y() = ced(rho, weight[6], {dirX.s5(), dirY.s5()}, u);
    e5678.z() = ced(rho, weight[7], {dirX.s6(), dirY.s6()}, u);
    e5678.w() = ced(rho, weight[8], {dirX.s7(), dirY.s7()}, u);

    e0    = (1.0 - omega) * f0    + omega * e0;
    e1234 = (1.0 - omega) * f1234 + omega * e1234;
    e5678 = (1.0 - omega) * f5678 + omega * e5678;
  }

  // Distribute the newly computed frequency distribution to neighbors
  bool t3 = idx > 0;          // Not on Left boundary
  bool t1 = idx < width - 1;  // Not on Right boundary
  bool t4 = idy > 0;          // Not on Upper boundary
  bool t2 = idy < height - 1; // Not on lower boundary

  if (t1 && t2 && t3 && t4) {
    // New positions to write (Each thread will write 8 values)
    // Note the propagation sources imply the OLD locations for each thread
    sycl::int8 nX = newPos(idx, dirX);
    sycl::int8 nY = newPos(idy, dirY);
    sycl::int8 nPos = nX + (sycl::int8)(width) * nY;

    // Write center distribution to thread's location
    of0[pos] = e0;

    // Propagate to right cell
    of1234[nPos.s0()].x() = e1234.x();

    // Propagate to Lower cell
    of1234[nPos.s1()].y() = e1234.y();

    // Propagate to left cell
    of1234[nPos.s2()].z() = e1234.z();

    // Propagate to Upper cell
    of1234[nPos.s3()].w() = e1234.w();

    // Propagate to Lower-Right cell
    of5678[nPos.s4()].x() = e5678.x();

    // Propogate to Lower-Left cell
    of5678[nPos.s5()].y() = e5678.y();

    // Propagate to Upper-Left cell
    of5678[nPos.s6()].z() = e5678.z();

    // Propagate to Upper-Right cell
    of5678[nPos.s7()].w() = e5678.w();
  }
}

void fluidSim (
  const int iterations,
  const double omega,
  const int *dims,
  const bool *h_type,
  sycl::double2 *u,
  double *rho,
  const sycl::double8 dirX,
  const sycl::double8 dirY,
  const double w[9],
        double *h_if0,
        double *h_if1234,
        double *h_if5678,
        double *h_of0,
        double *h_of1234,
        double *h_of5678)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int groupSize = GROUP_SIZE;
  size_t temp = dims[0] * dims[1];

  size_t dbl_size = temp * sizeof(double);
  size_t dbl4_size = temp * sizeof(sycl::double4);
  size_t bool_size = temp * sizeof(bool);

  // allocate and initialize device buffers
  double *d_if0 = sycl::malloc_device<double>(temp, q);
  q.memcpy(d_if0, h_if0, dbl_size);

  double *d_of0 = sycl::malloc_device<double>(temp, q);

  sycl::double4 *d_if1234 = sycl::malloc_device<sycl::double4>(temp, q);
  sycl::double4 *d_if5678 = sycl::malloc_device<sycl::double4>(temp, q);
  sycl::double4 *d_of1234 = sycl::malloc_device<sycl::double4>(temp, q);
  sycl::double4 *d_of5678 = sycl::malloc_device<sycl::double4>(temp, q);

  q.memcpy(d_of0, d_if0, dbl_size);
  q.memcpy(d_if1234, (sycl::double4*)h_if1234, dbl4_size);
  q.memcpy(d_if5678, (sycl::double4*)h_if5678, dbl4_size);
  q.memcpy(d_of1234, d_if1234, dbl4_size);
  q.memcpy(d_of5678, d_if5678, dbl4_size);

  // Constant bool array for position type = boundary or fluid
  bool *d_type = sycl::malloc_device<bool>(temp, q);
  q.memcpy(d_type, h_type, bool_size);

  // Weights for each distribution
  double *d_weight = sycl::malloc_device<double>(9, q);
  q.memcpy(d_weight, w, 9 * sizeof(double));

  sycl::range<2> lws (1, groupSize);
  sycl::range<2> gws (dims[1], dims[0]); 

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; ++i) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class fluidSim>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        lbm(item,
            d_if0,
            d_of0,
            d_if1234,
            d_of1234,
            d_if5678,
            d_of5678,
            d_type,
            dirX,
            dirY,
            d_weight,
            omega);
      });
    });

    // Swap device buffers
    auto temp0 = d_of0;
    auto temp1234 = d_of1234;
    auto temp5678 = d_of5678;

    d_of0 = d_if0;
    d_of1234 = d_if1234;
    d_of5678 = d_if5678;

    d_if0 = temp0;
    d_if1234 = temp1234;
    d_if5678 = temp5678;
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iterations);

  q.memcpy(h_of0, d_if0, dbl_size);
  q.memcpy((sycl::double4*)h_of1234, d_if1234, dbl4_size);
  q.memcpy((sycl::double4*)h_of5678, d_if5678, dbl4_size);
  q.wait();

  free(d_if0, q);
  free(d_of0, q);
  free(d_if1234, q);
  free(d_of1234, q);
  free(d_if5678, q);
  free(d_of5678, q);
  free(d_type, q);
  free(d_weight, q);
}
