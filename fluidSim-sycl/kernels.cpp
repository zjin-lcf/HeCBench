#include "common.h"

// Thread block size
#define GROUP_SIZE 256

// Calculates equilibrium distribution 
double ced(double rho, double weight, const double2 dir, const double2 u)
{
  double u2 = (u.x() * u.x()) + (u.y() * u.y());
  double eu = (dir.x() * u.x()) + (dir.y() * u.y());
  return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

// convert_int8() may be language specific
inline int8 newPos (const int p, const double8 &dir) {
  int8 np;
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
    nd_item<2> &item, 
    const double *__restrict if0,
          double *__restrict of0, 
    const double4 *__restrict if1234,
          double4 *__restrict of1234,
    const double4 *__restrict if5678,
          double4 *__restrict of5678,
    const bool *__restrict type,
    const double8 dirX,
    const double8 dirY,
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
  double4 f1234 = if1234[pos];
  double4 f5678 = if5678[pos];

  // intermediate results
  double e0;
  double4 e1234;
  double4 e5678;

  double rho; // Density
  double2 u;  // Velocity

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
    u = (double2)(0);
  }
  else // Fluid
  {
    // Compute Rho (density) of a cell by a reduction on f
    double4 temp = f1234 + f5678;
    temp.lo() += temp.hi();
    rho = f0 + temp.x() + temp.y();

    // Compute u (velocity) of a cell in x and y directions
    double4 x1234 = dirX.lo();
    double4 x5678 = dirX.hi();
    double4 y1234 = dirY.lo();
    double4 y5678 = dirY.hi();
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
    int8 nX = newPos(idx, dirX);
    int8 nY = newPos(idy, dirY);
    int8 nPos = nX + (int8)(width) * nY;

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
  double2 *u,
  double *rho,
  const double8 dirX,
  const double8 dirY,
  const double w[9],
        double *h_if0,
        double *h_if1234,
        double *h_if5678,
        double *h_of0,
        double *h_of1234,
        double *h_of5678)
{
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  int groupSize = GROUP_SIZE;
  size_t temp = dims[0] * dims[1];

  // allocate and initialize device buffers
  buffer<double, 1> d_if0 (h_if0, temp);
  d_if0.set_final_data(nullptr);
  buffer<double, 1> d_of0 (temp);
  buffer<double4, 1> d_if1234 (temp);
  buffer<double4, 1> d_if5678 (temp);
  buffer<double4, 1> d_of1234 (temp);
  buffer<double4, 1> d_of5678 (temp);

  q.submit([&] (handler &cgh) {
    auto in = d_if0.get_access<sycl_read>(cgh);
    auto out = d_of0.get_access<sycl_discard_write>(cgh);
    cgh.copy(in, out);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_if1234.get_access<sycl_discard_write>(cgh);
    cgh.copy((double4*)h_if1234, acc);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_if5678.get_access<sycl_discard_write>(cgh);
    cgh.copy((double4*)h_if5678, acc);
  });

  q.submit([&] (handler &cgh) {
    auto in = d_if1234.get_access<sycl_read>(cgh);
    auto out = d_of1234.get_access<sycl_discard_write>(cgh);
    cgh.copy(in, out);
  });

  q.submit([&] (handler &cgh) {
    auto in = d_if5678.get_access<sycl_read>(cgh);
    auto out = d_of5678.get_access<sycl_discard_write>(cgh);
    cgh.copy(in, out);
  });

  // Constant bool array for position type = boundary or fluid
  buffer<bool, 1> d_type (h_type, temp);

  // Weights for each distribution
  buffer<double, 1> d_weight (w, 9);

  range<2> lws (1, groupSize);
  range<2> gws (dims[1], dims[0]); 

  for(int i = 0; i < iterations; ++i) {
    q.submit([&] (handler &cgh) {
      auto if0 = d_if0.get_access<sycl_read>(cgh);
      auto of0 = d_of0.get_access<sycl_write>(cgh);
      auto if1234 = d_if1234.get_access<sycl_read>(cgh);
      auto of1234 = d_of1234.get_access<sycl_write>(cgh);
      auto if5678 = d_if5678.get_access<sycl_read>(cgh);
      auto of5678 = d_of5678.get_access<sycl_write>(cgh);
      auto w = d_weight.get_access<sycl_read>(cgh);
      auto t = d_type.get_access<sycl_read>(cgh);
      cgh.parallel_for<class fluidSim>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        lbm(item,
            if0.get_pointer(),
            of0.get_pointer(), 
            if1234.get_pointer(),
            of1234.get_pointer(),
            if5678.get_pointer(),
            of5678.get_pointer(),
            t.get_pointer(),
            dirX,
            dirY,
            w.get_pointer(),
            omega);
      });
    });

    // Swap device buffers
    auto temp0 = std::move(d_of0);
    auto temp1234 = std::move(d_of1234);
    auto temp5678 = std::move(d_of5678);

    d_of0 = std::move(d_if0);
    d_of1234 = std::move(d_if1234);
    d_of5678 = std::move(d_if5678);

    d_if0 = std::move(temp0);
    d_if1234 = std::move(temp1234);
    d_if5678 = std::move(temp5678);
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_if0.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_of0);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_if1234.get_access<sycl_read>(cgh);
    cgh.copy(acc, (double4*)h_of1234);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_if5678.get_access<sycl_read>(cgh);
    cgh.copy(acc, (double4*)h_of5678);
  });

  q.wait();
}
