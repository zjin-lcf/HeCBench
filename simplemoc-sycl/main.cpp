#include "SimpleMOC-kernel_header.h"
#include <sycl/sycl.hpp>

// host
void attenuate_segment( Input *I, Source *S,
    int QSR_id, int FAI_id, float *state_flux,
    SIMD_Vectors *simd_vecs)
{
  // Unload local vector vectors
  float *q0 =            simd_vecs->q0;
  float *q1 =            simd_vecs->q1;
  float *q2 =            simd_vecs->q2;
  float *sigT =          simd_vecs->sigT;
  float *tau =           simd_vecs->tau;
  float *sigT2 =         simd_vecs->sigT2;
  float *expVal =        simd_vecs->expVal;
  float *reuse =         simd_vecs->reuse;
  float *flux_integral = simd_vecs->flux_integral;
  float *tally =         simd_vecs->tally;
  float *t1 =            simd_vecs->t1;
  float *t2 =            simd_vecs->t2;
  float *t3 =            simd_vecs->t3;
  float *t4 =            simd_vecs->t4;

  // Some placeholder constants - In the full app some of these are
  // calculated based off position in geometry. This treatment
  // shaves off a few FLOPS, but is not significant compared to the
  // rest of the function.
  const float dz = 0.1f;
  const float zin = 0.3f;
  const float weight = 0.5f;
  const float mu = 0.9f;
  const float mu2 = 0.3f;
  const float ds = 0.7f;

  const int egroups = I->egroups;

  // load fine source region flux vector
  float * FSR_flux = &S[QSR_id].fine_flux[FAI_id * egroups];

  if( FAI_id == 0 )
  {
    float * f2 = &S[QSR_id].fine_source[FAI_id*egroups];
    float * f3 = &S[QSR_id].fine_source[(FAI_id+1)*egroups];
    // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
    for( int g = 0; g < egroups; g++)
    {
      // load neighboring sources
      const float y2 = f2[g];
      const float y3 = f3[g];

      // do linear "fitting"
      const float c0 = y2;
      const float c1 = (y3 - y2) / dz;

      // calculate q0, q1, q2
      q0[g] = c0 + c1*zin;
      q1[g] = c1;
      q2[g] = 0;
    }
  }
  else if ( FAI_id == I->fine_axial_intervals - 1 )
  {
    float * f1 = &S[QSR_id].fine_source[(FAI_id-1)*egroups];
    float * f2 = &S[QSR_id].fine_source[FAI_id*egroups];
    // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
    for( int g = 0; g < egroups; g++)
    {
      // load neighboring sources
      const float y1 = f1[g];
      const float y2 = f2[g];

      // do linear "fitting"
      const float c0 = y2;
      const float c1 = (y2 - y1) / dz;

      // calculate q0, q1, q2
      q0[g] = c0 + c1*zin;
      q1[g] = c1;
      q2[g] = 0;
    }
  }
  else
  {
    float * f1 = &S[QSR_id].fine_source[(FAI_id-1)*egroups];
    float * f2 = &S[QSR_id].fine_source[FAI_id*egroups];
    float * f3 = &S[QSR_id].fine_source[(FAI_id+1)*egroups];
    // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
    for( int g = 0; g < egroups; g++)
    {
      // load neighboring sources
      const float y1 = f1[g];
      const float y2 = f2[g];
      const float y3 = f3[g];

      // do quadratic "fitting"
      const float c0 = y2;
      const float c1 = (y1 - y3) / (2.f*dz);
      const float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

      // calculate q0, q1, q2
      q0[g] = c0 + c1*zin + c2*zin*zin;
      q1[g] = c1 + 2.f*c2*zin;
      q2[g] = c2;
    }
  }


  // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    // load total cross section
    sigT[g] = S[QSR_id].sigT[g];

    // calculate common values for efficiency
    tau[g] = sigT[g] * ds;
    sigT2[g] = sigT[g] * sigT[g];
  }


  // cycle over energy groups
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    expVal[g] = 1.f - expf( -tau[g] ); // exp is faster on many architectures
  }

  // Flux Integral

  // Re-used Term
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    reuse[g] = tau[g] * (tau[g] - 2.f) + 2.f * expVal[g]
      / (sigT[g] * sigT2[g]);
  }


  //#pragma vector alignednontemporal
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    // add contribution to new source flux
    flux_integral[g] = (q0[g] * tau[g] + (sigT[g] * state_flux[g] - q0[g])
        * expVal[g]) / sigT2[g] + q1[g] * mu * reuse[g] + q2[g] * mu2
      * (tau[g] * (tau[g] * (tau[g] - 3.f) + 6.f) - 6.f * expVal[g])
      / (3.f * sigT2[g] * sigT2[g]);
  }


#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    // Prepare tally
    tally[g] = weight * flux_integral[g];
  }

#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    FSR_flux[g] += tally[g];
  }


  // Term 1
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t1[g] = q0[g] * expVal[g] / sigT[g];
  }
  // Term 2
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t2[g] = q1[g] * mu * (tau[g] - expVal[g]) / sigT2[g];
  }
  // Term 3
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t3[g] =  q2[g] * mu2 * reuse[g];
  }
  // Term 4
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t4[g] = state_flux[g] * (1.f - expVal[g]);
  }

#ifdef VERIFY
  for( int g = 0; g < egroups; g++) {
    printf("q0[%d] = %f\n", g, q0[g]);
    printf("q1[%d] = %f\n", g, q1[g]);
    printf("q2[%d] = %f\n", g, q2[g]);
    printf("sigT[%d] = %f\n", g, sigT[g]);
    printf("tau[%d] = %f\n", g, tau[g]);
    printf("sigT2[%d] = %f\n", g, sigT2[g]);
    printf("expVal[%d] = %f\n", g, expVal[g]);
    printf("reuse[%d] = %f\n", g, reuse[g]);
    printf("flux_integral[%d] = %f\n", g, flux_integral[g]);
    printf("tally[%d] = %f\n", g, tally[g]);
    printf("t1[%d] = %f\n", g, t1[g]);
    printf("t2[%d] = %f\n", g, t2[g]);
    printf("t3[%d] = %f\n", g, t3[g]);
    printf("t4[%d] = %f\n", g, t4[g]);
  }
#endif

  // Total psi
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    state_flux[g] = t1[g] + t2[g] + t3[g] + t4[g];
  }
}


int main( int argc, char * argv[] )
{
  unsigned int seed = 2;

  srand(seed);

  // Get Inputs
  Input * I = set_default_input();
  read_CLI( argc, argv, I );

  // Calculate Number of 3D Source Regions
  I->source_3D_regions = (int) std::ceil((double)I->source_2D_regions *
      I->coarse_axial_intervals / I->decomp_assemblies_ax);

  logo(4); // Based on the 4th version

  // Build Source data (needed when verification is disabled)
  Source *S = initialize_sources(I);

  // Build Device data from Source data
  Source *S2 = copy_sources(I, S);

  print_input_summary(I);

  center_print("SIMULATION", 79);
  border_print();
  printf("Attentuating fluxes across segments...\n");

  // Run Simulation Kernel Loop

  // Host allocation
  SIMD_Vectors simd_vecs = allocate_simd_vectors(I);

  float * state_flux = (float *) malloc(I->egroups * sizeof(float));

  // Device allocation
  float * state_flux_device = NULL;
  posix_memalign( (void**)&state_flux_device, 1024, I->egroups * sizeof(float));

  int* QSR_id_arr = NULL;
  int* FAI_id_arr = NULL;
  posix_memalign( (void**)&QSR_id_arr, 1024, sizeof(int) * I->segments );
  posix_memalign( (void**)&FAI_id_arr, 1024, sizeof(int) * I->segments );

  // initialize the state flux
  for( int i = 0; i < I->egroups; i++ ) {
    state_flux_device[i] = rand_r(&seed) / (float) RAND_MAX;
    state_flux[i] = state_flux_device[i];
  }

  // Verification is performed for one segment;
  // Attenuate segment is not run on CPU to reduce simulation time
  for( long i = 0; i < I->segments; i++ )
  {
    // Pick Random QSR
    int QSR_id = rand_r(&seed) % I->source_3D_regions;

    // for device
    QSR_id_arr[i] = QSR_id;

    // Pick Random Fine Axial Interval
    int FAI_id = rand_r(&seed) % I->fine_axial_intervals;

    // for device
    FAI_id_arr[i] = FAI_id;

    // Attenuate Segment for one segment
#ifdef VERIFY
      attenuate_segment( I, S, QSR_id, FAI_id, state_flux, &simd_vecs);
#endif
  }

#ifdef VERIFY
  float* simd_vecs_debug = (float*) malloc (sizeof(float)*I->egroups*14);
#endif

  double kstart, kstop;
  double start = get_time();


#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_QSR_id = sycl::malloc_device<int>(I->segments, q);
  q.memcpy(d_QSR_id, QSR_id_arr, sizeof(int) * I->segments);

  int *d_FAI_id = sycl::malloc_device<int>(I->segments, q);
  q.memcpy(d_FAI_id, FAI_id_arr, sizeof(int) * I->segments);

  float *d_fine_flux = sycl::malloc_device<float>(
    I->source_3D_regions * I->fine_axial_intervals * I->egroups, q);
  q.memcpy(d_fine_flux, S2->fine_flux, sizeof(float) *
    I->source_3D_regions * I->fine_axial_intervals * I->egroups);

  float *d_fine_source = sycl::malloc_device<float>(
    I->source_3D_regions * I->fine_axial_intervals * I->egroups, q);

  q.memcpy(d_fine_source, S2->fine_source, sizeof(float) *
    I->source_3D_regions * I->fine_axial_intervals * I->egroups);

  float *d_sigT = sycl::malloc_device<float>(I->source_3D_regions * I->egroups, q);
  q.memcpy(d_sigT, S2->sigT, sizeof(float) *
    I->source_3D_regions * I->egroups);

  float *d_state_flux = sycl::malloc_device<float>(I->egroups, q);
  q.memcpy(d_state_flux, state_flux_device, sizeof(float) * I->egroups);

  float *d_simd_vecs = sycl::malloc_device<float>(I->egroups*14, q);

  int fine_axial_intervals = I->fine_axial_intervals;
  int egroups = I->egroups;
  int segments = I->segments;

  sycl::range<1> gws ((segments+127)/128*128);
  sycl::range<1> lws (128);

  kstart = get_time();

  for (int n = 0; n < I->repeat; n++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class attenuate_segment>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int gid = item.get_global_id(0);
        if (gid >= segments) return;

        const float dz = 0.1f;
        const float zin = 0.3f;
        const float weight = 0.5f;
        const float mu = 0.9f;
        const float mu2 = 0.3f;
        const float ds = 0.7f;

        int QSR_id = d_QSR_id[gid];
        int FAI_id = d_FAI_id[gid];

        // load fine source region flux vector
        int offset = QSR_id * fine_axial_intervals * egroups;

        float *FSR_flux = d_fine_flux +  offset + FAI_id * egroups;

        float* q0 = d_simd_vecs;
        float* q1 = d_simd_vecs + egroups;
        float* q2 = d_simd_vecs + egroups * 2;
        float* sigT = d_simd_vecs + egroups * 3;
        float* tau = d_simd_vecs + egroups * 4;
        float* sigT2 = d_simd_vecs + egroups * 5;
        float* expVal = d_simd_vecs + egroups * 6;
        float* reuse = d_simd_vecs + egroups * 7;
        float* flux_integral = d_simd_vecs + egroups * 8;
        float* tally = d_simd_vecs + egroups * 9;
        float* t1 = d_simd_vecs + egroups * 10;
        float* t2 = d_simd_vecs + egroups * 11;
        float* t3 = d_simd_vecs + egroups * 12;
        float* t4 = d_simd_vecs + egroups * 13;

        if( FAI_id == 0 )
        {
          float * f2 = d_fine_source + offset + FAI_id*egroups;
          float * f3 = d_fine_source + offset + (FAI_id+1)*egroups;
          // cycle over energy groups
          for( int g = 0; g < egroups; g++)
          {
            // load neighboring sources
            const float y2 = f2[g];
            const float y3 = f3[g];

            // do linear "fitting"
            const float c0 = y2;
            const float c1 = (y3 - y2) / dz;

            // calculate q0, q1, q2
            q0[g] = c0 + c1*zin;
            q1[g] = c1;
            q2[g] = 0;
          }
        }
        else if ( FAI_id == fine_axial_intervals - 1 )
        {
          float * f1 = d_fine_source + offset + (FAI_id-1)*egroups;
          float * f2 = d_fine_source + offset + FAI_id*egroups;

          for( int g = 0; g < egroups; g++)
          {
            // load neighboring sources
            const float y1 = f1[g];
            const float y2 = f2[g];

            // do linear "fitting"
            const float c0 = y2;
            const float c1 = (y2 - y1) / dz;

            // calculate q0, q1, q2
            q0[g] = c0 + c1*zin;
            q1[g] = c1;
            q2[g] = 0;
          }
        }
        else
        {
          float * f1 = d_fine_source + offset + (FAI_id-1)*egroups;
          float * f2 = d_fine_source + offset + FAI_id*egroups;
          float * f3 = d_fine_source + offset + (FAI_id+1)*egroups;
          // cycle over energy groups
          for( int g = 0; g < egroups; g++)
          {
            // load neighboring sources
            const float y1 = f1[g];
            const float y2 = f2[g];
            const float y3 = f3[g];

            // do quadratic "fitting"
            const float c0 = y2;
            const float c1 = (y1 - y3) / (2.f*dz);
            const float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

            // calculate q0, q1, q2
            q0[g] = c0 + c1*zin + c2*zin*zin;
            q1[g] = c1 + 2.f*c2*zin;
            q2[g] = c2;
          }
        }

        // cycle over energy groups
        offset = QSR_id * egroups;
        for( int g = 0; g < egroups; g++)
        {
          // load total cross section
          sigT[g] = d_sigT[offset + g];

          // calculate common values for efficiency
          tau[g] = sigT[g] * ds;
          sigT2[g] = sigT[g] * sigT[g];

          expVal[g] = 1.f - sycl::exp( -tau[g] ); // exp is faster on many architectures
          reuse[g] = tau[g] * (tau[g] - 2.f) + 2.f * expVal[g] / (sigT[g] * sigT2[g]);

          // add contribution to new source flux
          flux_integral[g] = (q0[g] * tau[g] + (sigT[g] * d_state_flux[g] - q0[g])
              * expVal[g]) / sigT2[g] + q1[g] * mu * reuse[g] + q2[g] * mu2
            * (tau[g] * (tau[g] * (tau[g] - 3.f) + 6.f) - 6.f * expVal[g])
            / (3.f * sigT2[g] * sigT2[g]);

          tally[g] = weight * flux_integral[g];
          FSR_flux[g] += tally[g];
          t1[g] = q0[g] * expVal[g] / sigT[g];
          t2[g] = q1[g] * mu * (tau[g] - expVal[g]) / sigT2[g];
          t3[g] = q2[g] * mu2 * reuse[g];
          t4[g] = d_state_flux[g] * (1.f - expVal[g]);
          d_state_flux[g] = t1[g]+t2[g]+t3[g]+t4[g];
        }
      });
    });
  }

  q.wait();
  kstop = get_time();

  q.memcpy(state_flux_device, d_state_flux, sizeof(float) * I->egroups);

  q.memcpy(S2->fine_flux, d_fine_flux, sizeof(float) *
    I->source_3D_regions * I->fine_axial_intervals * I->egroups);

#ifdef VERIFY
  q.memcpy(simd_vecs_debug, d_simd_vecs, sizeof(float) * I->egroups * 14);
#endif
  q.wait();

  sycl::free(d_QSR_id, q);
  sycl::free(d_FAI_id, q);
  sycl::free(d_fine_flux, q);
  sycl::free(d_fine_source, q);
  sycl::free(d_sigT, q);
  sycl::free(d_state_flux, q);
  sycl::free(d_simd_vecs, q);

  double stop = get_time();
  printf("Simulation Complete.\n");

#ifdef VERIFY
  int egroups = I->egroups;
  const float* q0 = simd_vecs_debug;
  const float* q1 = simd_vecs_debug + egroups;
  const float* q2 = simd_vecs_debug + egroups * 2;
  const float* sigT = simd_vecs_debug + egroups * 3;
  const float* tau = simd_vecs_debug + egroups * 4;
  const float* sigT2 = simd_vecs_debug + egroups * 5;
  const float* expVal = simd_vecs_debug + egroups * 6;
  const float* reuse = simd_vecs_debug + egroups * 7;
  const float* flux_integral = simd_vecs_debug + egroups * 8;
  const float* tally = simd_vecs_debug + egroups * 9;
  const float* t1 = simd_vecs_debug + egroups * 10;
  const float* t2 = simd_vecs_debug + egroups * 11;
  const float* t3 = simd_vecs_debug + egroups * 12;
  const float* t4 = simd_vecs_debug + egroups * 13;
  for (int g = 0; g < I->egroups; g++) {
    printf("q0[%d] = %f\n", g, q0[g]);
    printf("q1[%d] = %f\n", g, q1[g]);
    printf("q2[%d] = %f\n", g, q2[g]);
    printf("sigT[%d] = %f\n", g, sigT[g]);
    printf("tau[%d] = %f\n", g, tau[g]);
    printf("sigT2[%d] = %f\n", g, sigT2[g]);
    printf("expVal[%d] = %f\n", g, expVal[g]);
    printf("reuse[%d] = %f\n", g, reuse[g]);
    printf("flux_integral[%d] = %f\n", g, flux_integral[g]);
    printf("tally[%d] = %f\n", g, tally[g]);
    printf("t1[%d] = %f\n", g, t1[g]);
    printf("t2[%d] = %f\n", g, t2[g]);
    printf("t3[%d] = %f\n", g, t3[g]);
    printf("t4[%d] = %f\n", g, t4[g]);
  }

  bool error = false;
  for (int i = 0; i < I->egroups; i++) {
    if ( fabs(state_flux_device[i] - state_flux[i]) > 1e-1 ) {
      printf("%f %f\n", state_flux_device[i], state_flux[i]);
      error = true;
      break;
    }
  }
  if (error)
    printf("Fail\n");
  else
    printf("Success\n");

#endif

  border_print();
  center_print("RESULTS SUMMARY", 79);
  border_print();

  printf("%-25s%.3lf seconds\n", "Total kernel time:", kstop-kstart);
  printf("%-25s%.3lf seconds\n", "Device offload time:", stop-start);

  double tpi = ((double) (kstop - kstart) / (I->repeat) /
                (double)I->segments / (double) I->egroups) * 1.0e9;
  printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
  border_print();

  free(simd_vecs.q0);
  free(state_flux);
  free(QSR_id_arr);
  free(FAI_id_arr);
  free(state_flux_device);

  free(S2->fine_source);
  free(S2->fine_flux);
  free(S2->sigT);
  free(S2);
  free(S[0].fine_source);
  free(S[0].fine_flux);
  free(S[0].sigT);
  free(S);
  free(I);
  return 0;
}
