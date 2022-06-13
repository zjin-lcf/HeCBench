#include <string.h>
#include <chrono>
#include "common.h"

#ifndef dataType
#define dataType double
#endif

#include "CustomComplex.h"
#include "utils.h"
#include "kernel.h"

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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // Creating device versions of the data
  buffer<CustomComplex<dataType>, 1> d_aqsmtemp (aqsmtemp, aqsmtemp_size);
  buffer<CustomComplex<dataType>, 1> d_aqsntemp (aqsntemp, aqsntemp_size);
  buffer<CustomComplex<dataType>, 1> d_I_eps_array (I_eps_array, I_eps_array_size);
  buffer<CustomComplex<dataType>, 1> d_wtilde_array (wtilde_array, wtilde_array_size);

  buffer<dataType, 1> d_vcoul (vcoul, vcoul_size);
  buffer<dataType, 1> d_wx_array (wx_array, wx_array_size);
  buffer<dataType, 1> d_achtemp_re (achtemp_re_size);
  buffer<dataType, 1> d_achtemp_im (achtemp_im_size);

  buffer<int, 1> d_inv_igp_index (inv_igp_index, inv_igp_index_size);
  buffer<int, 1> d_indinv (indinv, indinv_size);

  // Time the kernel execution
  timeval startKernelTimer, endKernelTimer;
  gettimeofday(&startKernelTimer, NULL);

  range<3> gws(1, ngpown, 32 * number_bands);
  range<3> lws(1, 1, 32);
  printf("Launching a kernel with global work size = "
         "(%d,%d,%d), and local work size = (%d,%d,%d) \n",
         number_bands * 32, ngpown, 1, 32, 1, 1);

  float total_time = 0.f;

  for (int i = 0; i < 10; i++) {
    // Reset the atomic sums
    q.submit([&] (handler &cgh) {
      auto acc = d_achtemp_re.get_access<sycl_discard_write>(cgh);
      cgh.copy(achtemp_re, acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_achtemp_im.get_access<sycl_discard_write>(cgh);
      cgh.copy(achtemp_im, acc);
    });

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (handler &cgh) {
      auto igp = d_inv_igp_index.get_access<sycl_read>(cgh);
      auto ind = d_indinv.get_access<sycl_read>(cgh);
      auto wxa = d_wx_array.get_access<sycl_read>(cgh);
      auto wta = d_wtilde_array.get_access<sycl_read>(cgh);
      auto amt = d_aqsmtemp.get_access<sycl_read>(cgh);
      auto ant = d_aqsntemp.get_access<sycl_read>(cgh);
      auto eps = d_I_eps_array.get_access<sycl_read>(cgh);
      auto vc = d_vcoul.get_access<sycl_read>(cgh);
      auto re = d_achtemp_re.get_access<sycl_read_write>(cgh);
      auto im = d_achtemp_im.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class gpp_kernel>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {

        solver(item, number_bands, ngpown, ncouls, 
               igp.get_pointer(), ind.get_pointer(), wxa.get_pointer(),
               wta.get_pointer(), amt.get_pointer(), ant.get_pointer(),
               eps.get_pointer(), vc.get_pointer(), re.get_pointer(), im.get_pointer());
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / 10.f);

  q.submit([&] (handler &cgh) {
    auto acc = d_achtemp_re.get_access<sycl_read>(cgh);
    cgh.copy(acc, achtemp_re);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_achtemp_im.get_access<sycl_read>(cgh);
    cgh.copy(acc, achtemp_im);
  });

  q.wait();

  gettimeofday(&endKernelTimer, NULL);

  elapsedKernelTimer =
      (endKernelTimer.tv_sec - startKernelTimer.tv_sec) +
      1e-6 * (endKernelTimer.tv_usec - startKernelTimer.tv_usec);

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

  std::cout << "********** Kernel Time Taken **********= " << elapsedKernelTimer
            << " secs" << std::endl;
  std::cout << "********** Total Time Taken **********= " << elapsedTimer << " secs"
            << std::endl;

  return 0;
}
