// begin of accelerate_particles
void accelerate_particles( const int numTeams, const int numThreads,
                           Particle* p, const int n, const float kSofteningSquared, const float kG )
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < n; i++) {
    auto pi = p[i];
    RealType acc0 = pi.acc[0];
    RealType acc1 = pi.acc[1];
    RealType acc2 = pi.acc[2];

    for (int j = 0; j < n; j++) {
      RealType dx, dy, dz;
      RealType distance_sqr = 0.0f;
      RealType distance_inv = 0.0f;

      auto pj = p[j];
      dx = pj.pos[0] - pi.pos[0];  // 1flop
      dy = pj.pos[1] - pi.pos[1];  // 1flop
      dz = pj.pos[2] - pi.pos[2];  // 1flop

      distance_sqr =
        dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
      distance_inv = 1.0f / sqrtf(distance_sqr);       // 1div+1sqrt

      acc0 += dx * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
      acc1 += dy * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
      acc2 += dz * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
    }
    pi.acc[0] = acc0;
    pi.acc[1] = acc1;
    pi.acc[2] = acc2;
    p[i] = pi;
  }
}
// end of accelerate_particles

void update_particles(const int numTeams, const int numThreads,
                      Particle *__restrict__ p,
                      RealType *__restrict__ e,
                      const int n, RealType dt)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < n; i++) {
    auto pi = p[i];
    pi.vel[0] += pi.acc[0] * dt;  // 2flops
    pi.vel[1] += pi.acc[1] * dt;  // 2flops
    pi.vel[2] += pi.acc[2] * dt;  // 2flops

    pi.pos[0] += pi.vel[0] * dt;  // 2flops
    pi.pos[1] += pi.vel[1] * dt;  // 2flops
    pi.pos[2] += pi.vel[2] * dt;  // 2flops

    pi.acc[0] = 0.f;
    pi.acc[1] = 0.f;
    pi.acc[2] = 0.f;

    e[i] = pi.mass *
      (pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] +
       pi.vel[2] * pi.vel[2]);  // 7flops

    p[i] = pi;
  }
}

void accumulate_energy(RealType *e, const int n) 
{
  #pragma omp target 
  for (int i = 1; i < n; i++) e[0] += e[i];
}

