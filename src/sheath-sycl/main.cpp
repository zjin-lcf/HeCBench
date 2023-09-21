// https://www.particleincell.com/2016/cuda-pic/
// https://www.particleincell.com/wp-content/uploads/2016/02/sheath-gpu.cu


/* 1D sheath PIC simulation with SYCL */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>


/*constants*/
#define EPS_0 8.85418782e-12 // F/m, vacuum permittivity
#define K 1.38065e-23        // J/K, Boltzmann constant
#define ME 9.10938215e-31    // kg, electron mass
#define QE 1.602176565e-19   // C, elementary charge
#define AMU 1.660538921e-27  // kg, atomic mass unit
#define EV_TO_K 11604.52     // 1eV in Kelvin, QE/K

/*simulation parameters, these could come from an input file*/
#define PLASMA_DEN 1e16      // plasma density to load
#define NUM_IONS 500000      // number of ions
#define NUM_ELECTRONS 500000 // number of electrons
#define DX 1e-4              // cell spacing
#define NC 100               // number of cells
#define NUM_TS 1000          // number of time steps
#define DT 1e-11             // time step size
#define ELECTRON_TEMP 3.0    // electron temperature in eV
#define ION_TEMP 1.0         // ion temperature in eV

/*domain parameters, set here so can access from GPU*/
#define X0 0           /*mesh origin*/
#define XL NC* DX      /*domain length*/
#define XMAX (X0 + XL) /*domain max position*/

const int THREADS_PER_BLOCK = 256;
/* Data structure to hold domain information*/
struct Domain
{
  const int ni      = NC + 1; /*number of nodes*/
  const double x0   = X0;
  const double dx   = DX;
  const double xl   = XL;
  const double xmax = XMAX;

  /*data structures*/
  double* phi; /*potential*/
  double* ef;  /*electric field on the cpu*/
  double* rho; /*charge density*/

  float* ndi; /*ion density on the CPU*/
  float* nde; /*electron density on the CPU*/
};

/* Data structure for particle storage **/
struct Particle
{
  double x;   /*position*/
  double v;   /*velocity*/
  bool alive; /*flag to avoid removing particles*/
};

/* Data structure to hold species information*/
struct Species
{
  double mass;   /*particle mass in kg*/
  double charge; /*particle charge in Coulomb*/
  double spwt;   /*species specific weight*/

  int np;             /*number of particles*/
  int np_alloc;       /*size of the allocated data array*/
  Particle* part;     /*array holding particles on the CPU*/
};

/** FUNCTION PROTOTYPES **/
double rnd();
double SampleVel(double v_th);
void ScatterSpecies(sycl::queue &q, Species* species, Particle *species_part_gpu,
                    float* den, float *den_gpu, double &time);
void ComputeRho(Species* ions, Species* electrons);
bool SolvePotential(double* phi, double* rho);
bool SolvePotentialDirect(double* phi, double* rho);
void ComputeEF(sycl::queue &q, double* phi, double* ef, double *ef_gpu);
void PushSpecies(sycl::queue &q, Species* species, Particle *species_part_gpu, double *ef_gpu);
void RewindSpecies(sycl::queue &q, Species* species, Particle *species_part_gpu, double *ef_gpu);
void AddParticle(Species* species, double x, double v);
double XtoL(double pos);
double gather(double lc, const double* field);
void scatter(double lc, float value, float* field);

void WriteResults(int ts);

/* GLOBAL VARIABLES */
Domain domain;

FILE* file_res;

/* --------- main -------------*/
int main(int argc, char* argv[])
{
  int p;
  int ts; // time step
  double sp_time = 0.0; // total time of the scatter-particle kernel

  domain.phi = new double[domain.ni]; // potential
  domain.rho = new double[domain.ni]; // charge density
  domain.ef  = new double[domain.ni]; // electric field
  domain.nde = new float[domain.ni];  // number density of electrons
  domain.ndi = new float[domain.ni];  // number density of ions

  /*save pointers so we can write phi instead of domain.phi*/
  double* phi = domain.phi;
  double* rho = domain.rho;
  double* ef  = domain.ef;
  float* nde  = domain.nde;
  float* ndi  = domain.ndi;

  /*clear data*/
  memset(phi, 0, sizeof(double) * domain.ni);

  /*variables to hold species data ions*/
  Species ions;
  Species electrons;

  /*set material data*/
  ions.mass     = 16 * AMU;
  ions.charge   = QE;
  ions.spwt     = PLASMA_DEN * domain.xl / NUM_IONS;
  ions.np       = 0;
  ions.np_alloc = NUM_IONS;
  ions.part     = new Particle[NUM_IONS];

  electrons.mass     = ME; // electrons
  electrons.charge   = -QE;
  electrons.spwt     = PLASMA_DEN * domain.xl / NUM_ELECTRONS;
  electrons.np       = 0;
  electrons.np_alloc = NUM_ELECTRONS;
  electrons.part     = new Particle[NUM_ELECTRONS];

  /*randomize RNG*/
  srand(123);

  /*load uniformly spaced ions */
  double delta_ions = domain.xl / NUM_IONS;
  double v_thi      = sqrt(2 * K * ION_TEMP * EV_TO_K / ions.mass);
  for (p = 0; p < NUM_IONS; p++)
  {
    double x = domain.x0 + p * delta_ions;
    double v = SampleVel(v_thi);
    AddParticle(&ions, x, v);
  }

  /*load uniformly spaced electrons */
  double delta_electrons = domain.xl / NUM_ELECTRONS;
  double v_the           = sqrt(2 * K * ELECTRON_TEMP * EV_TO_K / electrons.mass);
  for (p = 0; p < NUM_ELECTRONS; p++)
  {
    double x = domain.x0 + p * delta_electrons;
    double v = SampleVel(v_the);
    AddParticle(&electrons, x, v);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  /*also allocate device memory */
  float *nde_gpu = sycl::malloc_device<float>(domain.ni, q);
  float *ndi_gpu = sycl::malloc_device<float>(domain.ni, q);
  double *ef_gpu = sycl::malloc_device<double>(domain.ni, q);
  Particle *ions_part_gpu = sycl::malloc_device<Particle>(NUM_IONS, q);
  q.memcpy(ions_part_gpu, ions.part, NUM_IONS * sizeof(Particle)).wait();

  Particle *electrons_part_gpu = sycl::malloc_device<Particle>(NUM_ELECTRONS, q);
  q.memcpy(electrons_part_gpu, electrons.part, NUM_ELECTRONS * sizeof(Particle)).wait();

  /*compute number density*/
  ScatterSpecies(q, &ions, ions_part_gpu, ndi, ndi_gpu, sp_time);
  ScatterSpecies(q, &electrons, electrons_part_gpu, nde, nde_gpu, sp_time);

  /*compute charge density and solve potential*/
  ComputeRho(&ions, &electrons);

  SolvePotential(phi, rho);

  ComputeEF(q, phi, ef, ef_gpu);

  RewindSpecies(q, &ions, ions_part_gpu, ef_gpu);
  RewindSpecies(q, &electrons, electrons_part_gpu, ef_gpu);

  /*OUTPUT*/
  file_res = fopen("result.dat", "w");
  fprintf(file_res, "VARIABLES = x nde ndi rho phi ef\n");
  WriteResults(0);

  auto start = std::chrono::steady_clock::now();

  /* MAIN LOOP*/
  for (ts = 1; ts <= NUM_TS; ts++)
  {
    /*compute number density*/
    ScatterSpecies(q, &ions, ions_part_gpu, ndi, ndi_gpu, sp_time);
    ScatterSpecies(q, &electrons, electrons_part_gpu, nde, nde_gpu, sp_time);

    ComputeRho(&ions, &electrons);
    SolvePotential(phi, rho);
    ComputeEF(q, phi, ef, ef_gpu);

    /*move particles*/
    PushSpecies(q, &electrons, electrons_part_gpu, ef_gpu);
    PushSpecies(q, &ions, ions_part_gpu, ef_gpu);

    /*write diagnostics*/
    if (ts % 25 == 0)
    {
      /*max phi*/
      double max_phi = abs(phi[0]);
      for (int i = 0; i < domain.ni; i++)
        if (abs(phi[i]) > max_phi)
          max_phi = abs(phi[i]);

      printf("TS:%i\tnp_i:%d\tnp_e:%d\tdphi:%.3g\n", ts, ions.np, electrons.np,
          max_phi - phi[0]);
    }

    /*save data*/
    if (ts % 1000 == 0)
      WriteResults(ts);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  fclose(file_res);

  /*free up memory*/
  delete phi;
  delete rho;
  delete ef;
  delete nde;
  delete ndi;
  sycl::free(nde_gpu, q);
  sycl::free(ndi_gpu, q);
  sycl::free(ef_gpu, q);

  /*free particles*/
  delete ions.part;
  delete electrons.part;
  sycl::free(ions_part_gpu, q);
  sycl::free(electrons_part_gpu, q);

  printf("Total kernel execution time (scatter particles) : %.3g (s)\n", sp_time * 1e-9f),
  printf("Total time for %d time steps: %.3g (s)\n", NUM_TS, time * 1e-9f);
  printf("Time per time step: %.3g (ms)\n", (time * 1e-6f) / NUM_TS);

  return 0;
}

/***** HELPER FUNCTIONS *********************************************************/
/* random number generator
   for now using built-in but this is not adequate for real simulations*/
double rnd()
{
  return rand() / (double)RAND_MAX;
}

/* samples random velocity from Maxwellian distribution using Birdsall's method*/
double SampleVel(double v_th)
{
  const int M = 12;
  double sum  = 0;
  for (int i = 0; i < M; i++)
    sum += rnd();

  return sqrt(0.5) * v_th * (sum - M / 2.0) / sqrt(M / 12.0);
}


/*scatter particles of species to the mesh*/
void ScatterSpecies(sycl::queue &q,
                    Species* species,
                    Particle *species_part_gpu,
                    float* den,
                    float *den_gpu,
                    double &time)
{
  /*initialize densities to zero*/
  q.memset(den_gpu, 0, sizeof(float) * domain.ni);

  int size = species->np_alloc;

  /*scatter particles to the mesh*/
  int nblocks = 1 + size / THREADS_PER_BLOCK;

  sycl::range<1> gws (nblocks * THREADS_PER_BLOCK);
  sycl::range<1> lws (THREADS_PER_BLOCK);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class scatterParticle>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      long p = item.get_global_id(0);
      if (p < size && species_part_gpu[p].alive)
      {
        double lc = XtoL(species_part_gpu[p].x);
        scatter(lc, 1.f, den_gpu);
      }
    });
  }).wait();

  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  /*copy density back to CPU*/
  q.memcpy(den, den_gpu, sizeof(float) * domain.ni).wait();

  /*divide by cell volume*/
  for (int i = 0; i < domain.ni; i++)
    den[i] *= species->spwt / domain.dx;

  /*only half cell at boundaries*/
  den[0] *= 2.0;
  den[domain.ni - 1] *= 2.0;
}

/*adds new particle to the species, returns pointer to the newly added data*/
void AddParticle(Species* species, double x, double v)
{
  /*abort the simulation if we ran out of space to store this particle*/
  if (species->np > species->np_alloc - 1)
  {
    printf("Too many particles!\n");
    exit(-1);
  }

  /*store position and velocity of this particle*/
  species->part[species->np].x     = x;
  species->part[species->np].v     = v;
  species->part[species->np].alive = true;

  /*increment particle counter*/
  species->np++;
}

/*computes charge density by adding ion and electron data*/
void ComputeRho(Species* ions, Species* electrons)
{
  double* rho = domain.rho;

  for (int i = 0; i < domain.ni; i++)
    rho[i] = ions->charge * domain.ndi[i] + electrons->charge * domain.nde[i];
}

/*Thomas algorithm for a tri-diagonal matrix*/
bool SolvePotentialDirect(double* x, double* rho)
{
  /*set coefficients, this should be pre-computed*/
  int ni     = domain.ni;
  double dx2 = domain.dx * domain.dx;
  int i;
  double* a = new double[ni];
  double* b = new double[ni];
  double* c = new double[ni];

  /*central difference on internal nodes*/
  for (i = 1; i < ni - 1; i++)
  {
    a[i] = 1;
    b[i] = -2;
    c[i] = 1;
  }

  /*dirichlet b.c. on boundaries*/
  a[0]      = 0;
  b[0]      = 1;
  c[0]      = 0;
  a[ni - 1] = 0;
  b[ni - 1] = 1;
  c[ni - 1] = 0;

  /*multiply RHS*/
  for (i = 1; i < domain.ni - 1; i++)
    x[i] = -rho[i] * dx2 / EPS_0;

  x[0]      = 0;
  x[ni - 1] = 0;

  /* Modify the coefficients. */
  c[0] /= b[0]; /* Division by zero risk. */
  x[0] /= b[0]; /* Division by zero would imply a singular matrix. */
  for (i = 1; i < ni; i++)
  {
    double id = (b[i] - c[i - 1] * a[i]); /* Division by zero risk. */
    c[i] /= id;                           /* Last value calculated is redundant. */
    x[i] = (x[i] - x[i - 1] * a[i]) / id;
  }

  /* Now back substitute. */
  for (i = ni - 2; i >= 0; i--)
    x[i] = x[i] - c[i] * x[i + 1];

  return true;
}

/* solves potential using the Gauss Seidel Method, returns true if converged*/
bool SolvePotential(double* phi, double* rho)
{
  double L2;
  double dx2 = domain.dx * domain.dx; /*precompute*/

  /*initialize boundaries*/
  phi[0] = phi[domain.ni - 1] = 0;

  /*solve potential, identical to lesson 2*/
  for (int solver_it = 0; solver_it < 40000; solver_it++)
  {
    /*Gauss Seidel method, phi[i-1]-2*phi[i]+phi[i+1] = -dx^2*rho[i]/eps_0*/
    for (int i = 1; i < domain.ni - 1; i++)
    {
      /*SOR*/
      double g = 0.5 * (phi[i - 1] + phi[i + 1] + dx2 * rho[i] / EPS_0);
      phi[i]   = phi[i] + 1.4 * (g - phi[i]);
    }

    /*check for convergence*/
    if (solver_it % 25 == 0)
    {
      double sum = 0;
      for (int i = 1; i < domain.ni - 1; i++)
      {
        double R = -rho[i] / EPS_0 - (phi[i - 1] - 2 * phi[i] + phi[i + 1]) / dx2;
        sum += R * R;
      }
      L2 = sqrt(sum) / domain.ni;
      if (L2 < 1e-4)
      {
        return true;
      }
    }
  }
  printf("Gauss-Seidel solver failed to converge, L2=%.3g!\n", L2);
  return false;
}

/* computes electric field by differentiating potential*/
void ComputeEF(sycl::queue &q, double* phi, double* ef, double *ef_gpu)
{
  for (int i = 1; i < domain.ni - 1; i++)
    ef[i] = -(phi[i + 1] - phi[i - 1]) / (2 * domain.dx); // central difference

  /*one sided difference at boundaries*/
  ef[0]             = -(phi[1] - phi[0]) / domain.dx;
  ef[domain.ni - 1] = -(phi[domain.ni - 1] - phi[domain.ni - 2]) / domain.dx;

  /*copy to the gpu*/
  q.memcpy(ef_gpu, ef, domain.ni * sizeof(double));
}

/* moves particles of a single species, returns wall charge*/
void PushSpecies(sycl::queue &q,
                 Species* species,
                 Particle *species_part_gpu,
                 double *ef_gpu)
{
  /*precompute q/m*/
  double qm = species->charge / species->mass;

  int size = species->np_alloc;

  /*loop over particles*/
  int nblocks = 1 + size / THREADS_PER_BLOCK;

  sycl::range<1> gws (nblocks * THREADS_PER_BLOCK);
  sycl::range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class pushParticle>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      /*get particle id*/
      long p = item.get_global_id(0);

      if (p < size && species_part_gpu[p].alive)
      {
        /*grab pointer to this particle*/
        Particle* part = &species_part_gpu[p];

        /*compute particle node position*/
        double lc = XtoL(part->x);

        /*gather electric field onto particle position*/
        double part_ef = gather(lc, ef_gpu);

        /*advance velocity*/
        part->v += DT * qm * part_ef;

        /*advance position*/
        part->x += DT * part->v;

        /*remove species_part_gpu leaving the domain*/
        if (part->x < X0 || part->x >= XMAX)
          part->alive = false;
      }
    });
  });
}


/* rewinds particle velocities by -0.5DT*/
void RewindSpecies(sycl::queue &q, Species* species, Particle *species_part_gpu, double *ef_gpu)
{
  /*precompute q/m*/
  double qm = species->charge / species->mass;

  int size = species->np_alloc;

  /*loop over particles*/
  int nblocks = 1 + size / THREADS_PER_BLOCK;

  sycl::range<1> gws (nblocks * THREADS_PER_BLOCK);
  sycl::range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class rewindParticle>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      /*get particle id*/
      long p = item.get_global_id(0);

      if (p < size && species_part_gpu[p].alive)
      {
        /*grab pointer to this particle*/
        Particle* part = &species_part_gpu[p];

        /*compute particle node position*/
        double lc = XtoL(part->x);

        /*gather electric field onto particle position*/
        double part_ef = gather(lc, ef_gpu);

        /*advance velocity*/
        part->v -= 0.5 * DT * qm * part_ef;
      }
    });
  });
}


/* converts physical coordinate to logical*/
double XtoL(double pos)
{
  double li = (pos - 0) / DX;
  return li;
}


/* atomic scatter of scalar value onto a field at logical coordinate lc*/
void scatter(double lc, float value, float* field)
{
  int i    = (int)lc;
  float di = lc - i;

  auto f1 = sycl::atomic_ref<float,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> (field[i]);
  f1.fetch_add(value * (1 - di));
  auto f2 = sycl::atomic_ref<float,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> (field[i+1]);
  f2.fetch_add(value * di);
}

/* gathers field value at logical coordinate lc*/
double gather(double lc, const double* field)
{
  int i     = (int)lc;
  double di = lc - i;

  /*gather field value onto particle position*/
  double val = field[i] * (1 - di) + field[i + 1] * (di);
  return val;
}


/* writes new zone to the results file*/
void WriteResults(int ts)
{
  fprintf(file_res, "ZONE I=%d T=ZONE_%06d\n", domain.ni, ts);
  for (int i = 0; i < domain.ni; i++)
  {
    fprintf(file_res, "%g %g %g %g %g %g\n", i * domain.dx, domain.nde[i], domain.ndi[i],
        domain.rho[i], domain.phi[i], domain.ef[i]);
  }

  fflush(file_res);
}
