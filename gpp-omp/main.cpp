#include <omp.h>
#include <chrono>
#include <string.h>

#ifndef dataType
#define dataType double
#endif

#include "CustomComplex.h"
#include "utils.h"

int main(int argc, char **argv) {

  int number_bands = 0, nvband = 0, ncouls = 0, nodes_per_group = 0;
  if (argc == 1) {
    number_bands = 512;
    nvband = 2;
    ncouls = 512;
    nodes_per_group = 20;
  } else if (argc == 2) {
    if (strcmp(argv[1], "benchmark") == 0) {
      number_bands = 512;
      nvband = 2;
      ncouls = 32768;
      nodes_per_group = 20;
    } else if (strcmp(argv[1], "test") == 0) {
      number_bands = 512;
      nvband = 2;
      ncouls = 512;
      nodes_per_group = 20;
    } else {
      std::cout
          << "Usage: ./main <test or benchmark>\n"
          << "Problem unrecognized, use 'test' or 'benchmark'"
          << std::endl;
      exit(0);
    }
  } else if (argc == 5) {
    number_bands = atoi(argv[1]);
    nvband = atoi(argv[2]);
    ncouls = atoi(argv[3]);
    nodes_per_group = atoi(argv[4]);
  } else {
    std::cout << "The correct form of input is : " << std::endl;
    std::cout << " ./main <number_bands> <number_valence_bands> "
                 "<number_plane_waves> <nodes_per_mpi_group> "
              << std::endl;
    exit(0);
  }
  int ngpown = ncouls / nodes_per_group;

  // Constants that will be used later
  const dataType e_lk = 10;
  const dataType dw = 1;
  const dataType to1 = 1e-6;
  const dataType e_n1kq = 6.0;

  // Start the timer before the work begins.
  dataType elapsedKernelTimer, elapsedTimer;
  timeval startTimer, endTimer;
  gettimeofday(&startTimer, NULL);

  // Printing out the params passed.
  std::cout << "Sizeof(CustomComplex<dataType> = "
            << sizeof(CustomComplex<dataType>) << " bytes" << std::endl;
  std::cout << "number_bands = " << number_bands << "\t nvband = " << nvband
            << "\t ncouls = " << ncouls
            << "\t nodes_per_group  = " << nodes_per_group
            << "\t ngpown = " << ngpown << "\t nend = " << nend
            << "\t nstart = " << nstart << std::endl;

  CustomComplex<dataType> expr0(0.0, 0.0);
  CustomComplex<dataType> expr(0.025, 0.025);
  size_t memFootPrint = 0;

  CustomComplex<dataType> *achtemp;
  achtemp = (CustomComplex<dataType> *)safe_malloc(
      achtemp_size * sizeof(CustomComplex<dataType>));

  memFootPrint += achtemp_size * sizeof(CustomComplex<dataType>);

  CustomComplex<dataType> *aqsmtemp, *aqsntemp;
  aqsmtemp = (CustomComplex<dataType> *)safe_malloc(
      aqsmtemp_size * sizeof(CustomComplex<dataType>));

  aqsntemp = (CustomComplex<dataType> *)safe_malloc(
      aqsntemp_size * sizeof(CustomComplex<dataType>));

  memFootPrint += 2 * aqsmtemp_size * sizeof(CustomComplex<dataType>);

  CustomComplex<dataType> *I_eps_array, *wtilde_array;
  I_eps_array = (CustomComplex<dataType> *)safe_malloc(
      I_eps_array_size * sizeof(CustomComplex<dataType>));

  wtilde_array = (CustomComplex<dataType> *)safe_malloc(
      I_eps_array_size * sizeof(CustomComplex<dataType>));

  memFootPrint += 2 * I_eps_array_size * sizeof(CustomComplex<dataType>);

  dataType *vcoul;
  vcoul = (dataType *)safe_malloc(vcoul_size * sizeof(dataType));

  memFootPrint += vcoul_size * sizeof(dataType);

  int *inv_igp_index, *indinv;
  inv_igp_index = (int *)safe_malloc(inv_igp_index_size * sizeof(int));
  indinv = (int *)safe_malloc(indinv_size * sizeof(int));

  // Real and imaginary parts of achtemp calculated separately
  dataType *achtemp_re, *achtemp_im, *wx_array;
  achtemp_re = (dataType *)safe_malloc(achtemp_re_size * sizeof(dataType));
  achtemp_im = (dataType *)safe_malloc(achtemp_im_size * sizeof(dataType));
  wx_array = (dataType *)safe_malloc(wx_array_size * sizeof(dataType));

  memFootPrint += 3 * wx_array_size * sizeof(double);

  // Memory footprint
  std::cout << "Memory Foot Print = " << memFootPrint / pow(1024, 3) << " GBs"
            << std::endl;

  for (int n1 = 0; n1 < number_bands; n1++)
    for (int ig = 0; ig < ncouls; ig++) {
      aqsmtemp(n1, ig) = expr;
      aqsntemp(n1, ig) = expr;
    }

  for (int my_igp = 0; my_igp < ngpown; my_igp++)
    for (int ig = 0; ig < ncouls; ig++) {
      I_eps_array(my_igp, ig) = expr;
      wtilde_array(my_igp, ig) = expr;
    }

  for (int i = 0; i < ncouls; i++)
    vcoul[i] = i * 0.025;

  for (int ig = 0; ig < ngpown; ++ig)
    inv_igp_index[ig] = (ig + 1) * ncouls / ngpown;

  for (int ig = 0; ig < ncouls; ++ig)
    indinv[ig] = ig;
  indinv[ncouls] = ncouls - 1;

  for (int iw = nstart; iw < nend; ++iw) {
    achtemp_re[iw] = 0.0;
    achtemp_im[iw] = 0.0;
  }

  for (int iw = nstart; iw < nend; ++iw) {
    wx_array[iw] = e_lk - e_n1kq + dw * ((iw + 1) - 2);
    if (wx_array[iw] < to1)
      wx_array[iw] = to1;
  }

  #pragma omp target data map(to \
    : aqsmtemp [0:aqsmtemp_size], vcoul [0:vcoul_size],                        \
      inv_igp_index [0:inv_igp_index_size], indinv [0:indinv_size],            \
      aqsntemp [0:aqsntemp_size], I_eps_array [0:I_eps_array_size],            \
      wx_array [0:wx_array_size], wtilde_array [0:wtilde_array_size])
  {
    dataType ach_re0, ach_re1, ach_re2, ach_im0, ach_im1, ach_im2;

    float total_time = 0.f;

    for (int i = 0; i < 10; i++) {
      ach_re0 = 0.0, ach_re1 = 0.0, ach_re2 = 0.0,
      ach_im0 = 0.0, ach_im1 = 0.0, ach_im2 = 0.0;

      auto start = std::chrono::steady_clock::now();

      #pragma omp target teams distribute parallel for collapse(2) \
       reduction(+:ach_re0, ach_re1, ach_re2, ach_im0, ach_im1, ach_im2)
      for (int my_igp = 0; my_igp < ngpown; ++my_igp) // 1634
      {
        for (int n1 = 0; n1 < number_bands; ++n1) // 512
        {
          int indigp = inv_igp_index[my_igp];
          int igp = indinv[indigp];

          dataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
          #pragma unroll
          for (size_t iw = nstart; iw < nend; ++iw) {
            achtemp_re_loc[iw] = 0.0;
            achtemp_im_loc[iw] = 0.0;
          }

          CustomComplex<dataType> sch_store1 =
              aqsmtemp(n1, igp).conj() * aqsntemp(n1, igp) * 0.5 *
              vcoul[igp];

          for (size_t ig = 0; ig < ncouls; ++ig) // 32768
          {
            #pragma unroll
            for (size_t iw = nstart; iw < nend; ++iw) // 3
            {
              CustomComplex<dataType> wdiff =
                  wx_array[iw] - wtilde_array(my_igp, ig);
              CustomComplex<dataType> delw =
                  wtilde_array(my_igp, ig) * wdiff.conj() *
                  (1 / CustomComplex_real((wdiff * wdiff.conj())));
              CustomComplex<dataType> sch_array =
                  delw * I_eps_array(my_igp, ig) * sch_store1;

              achtemp_re_loc[iw] += sch_array.real();
              achtemp_im_loc[iw] += sch_array.imag();
            }
          }

          ach_re0 += achtemp_re_loc[0];
          ach_re1 += achtemp_re_loc[1];
          ach_re2 += achtemp_re_loc[2];
          ach_im0 += achtemp_im_loc[0];
          ach_im1 += achtemp_im_loc[1];
          ach_im2 += achtemp_im_loc[2];
        } // ngpown
      }   // number_bands

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      total_time += time;
    } // for

    printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / 10.f);

    achtemp_re[0] = ach_re0;
    achtemp_re[1] = ach_re1;
    achtemp_re[2] = ach_re2;
    achtemp_im[0] = ach_im0;
    achtemp_im[1] = ach_im1;
    achtemp_im[2] = ach_im2;

    for (int iw = nstart; iw < nend; ++iw)
      achtemp[iw] = CustomComplex<dataType>(achtemp_re[iw], achtemp_im[iw]);

    // Check for correctness
    if (argc == 2) {
      if (strcmp(argv[1], "benchmark") == 0)
        correctness(0, achtemp[0]);
      else if (strcmp(argv[1], "test") == 0)
        correctness(1, achtemp[0]);
    } else
      correctness(1, achtemp[0]);

    printf("\n Final achtemp\n");
    achtemp[0].print();

    gettimeofday(&endTimer, NULL);
    elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +
                   1e-6 * (endTimer.tv_usec - startTimer.tv_usec);
  }

  // Free the allocated memory
  free(achtemp);
  free(aqsmtemp);
  free(aqsntemp);
  free(I_eps_array);
  free(wtilde_array);
  free(vcoul);
  free(inv_igp_index);
  free(indinv);
  free(achtemp_re);
  free(achtemp_im);
  free(wx_array);

  std::cout << "********** Total Time Taken **********= " << elapsedTimer << " secs"
            << std::endl;

  return 0;
}
