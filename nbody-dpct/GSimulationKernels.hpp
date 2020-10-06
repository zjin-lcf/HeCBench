#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void accelerate_particles(Particle *p, const int n,
                          const float kSofteningSquared, const float kG,
                          sycl::nd_item<3> item_ct1)
{
  int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
  RealType acc0 = p[i].acc[0];
  RealType acc1 = p[i].acc[1];
  RealType acc2 = p[i].acc[2];
  for (int j = 0; j < n; j++) {
    RealType dx, dy, dz;
    RealType distance_sqr = 0.0f;
    RealType distance_inv = 0.0f;

    dx = p[j].pos[0] - p[i].pos[0];  // 1flop
    dy = p[j].pos[1] - p[i].pos[1];  // 1flop
    dz = p[j].pos[2] - p[i].pos[2];  // 1flop

    distance_sqr =
      dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
    distance_inv = 1.0f / sycl::sqrt((float)distance_sqr); // 1div+1sqrt

    acc0 += dx * kG * p[j].mass * distance_inv * distance_inv *
      distance_inv;  // 6flops
    acc1 += dy * kG * p[j].mass * distance_inv * distance_inv *
      distance_inv;  // 6flops
    acc2 += dz * kG * p[j].mass * distance_inv * distance_inv *
      distance_inv;  // 6flops
  }
  p[i].acc[0] = acc0;
  p[i].acc[1] = acc1;
  p[i].acc[2] = acc2;
}

void
update_particles( Particle* p, RealType* e, const int n, RealType dt,
                  sycl::nd_item<3> item_ct1)
{
  int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
  p[i].vel[0] += p[i].acc[0] * dt;  // 2flops
  p[i].vel[1] += p[i].acc[1] * dt;  // 2flops
  p[i].vel[2] += p[i].acc[2] * dt;  // 2flops

  p[i].pos[0] += p[i].vel[0] * dt;  // 2flops
  p[i].pos[1] += p[i].vel[1] * dt;  // 2flops
  p[i].pos[2] += p[i].vel[2] * dt;  // 2flops

  p[i].acc[0] = 0.f;
  p[i].acc[1] = 0.f;
  p[i].acc[2] = 0.f;

  e[i] = p[i].mass *
    (p[i].vel[0] * p[i].vel[0] + p[i].vel[1] * p[i].vel[1] +
     p[i].vel[2] * p[i].vel[2]);  // 7flops
}

void
accumulate_energy(RealType *e, const int n) 
{
  for (int i = 1; i < n; i++) e[0] += e[i];
}

