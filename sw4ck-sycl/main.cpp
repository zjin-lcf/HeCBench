//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2021, Lawrence Livermore National Security, LLC and SW4CK
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: GPL-2.0-only
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <tuple>
#include <chrono>
#include <limits>
#include <cmath>
#include <sycl/sycl.hpp>

#include "utils.h"
#include "utils.cpp"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to file> <repeat>\n";
    return 1;
  }

  // Open an input data file
  std::ifstream iff;
  iff.open(argv[1]);

  // Repeat the execution of kernels
  const int repeat = atoi(argv[2]);

  // At most 10 input datasets
  std::map<std::string, Sarray*> arrays[10];
  std::vector<int*> onesided;
  std::string line;
  int lc = 0;
  std::cout << "Reading from file " << argv[1] << "\n";
  while (std::getline(iff, line)) {
    std::istringstream iss(line);
    int* optr = new int[14];
    const int N = 16;
    if ((lc % N) == 0) {
      if (!(iss >> optr[0] >> optr[1] >> optr[2] >> optr[3] >> optr[4] >>
            optr[5] >> optr[6] >> optr[7] >> optr[8] >> optr[9] >> optr[10] >>
            optr[11] >> optr[12] >> optr[13])) {
        std::cerr << "Error reading data on line " << lc + 1 << "\n";
        break;
      }
      onesided.push_back(optr);
    } else {
      Sarray* s = new Sarray();
      auto name = s->fill(iss);
      if (name == "Break") {
        std::cerr << "Error reading Sarray data on line " << lc + 1 << "\n";
        break;
      } else {
        arrays[lc / N][name] = s;
      }
    }
    lc++;
  } // while

#ifdef VERBOSE
  std::cout << "\nCurrent state of map array\n";
#endif
  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i]) {
#ifdef VERBOSE
      std::cout << x.first << " " << x.second->g << " " << x.second->m_npts
        << "\n";
#endif
      x.second->init();
    }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  //
  // Allocate device memory explictly
  //
  int size = (6 + 384 + 24 + 48 + 6 + 384 + 6 + 6);
  float_sw4 *cof_ptr = (float_sw4*) malloc (sizeof(float_sw4) * size);
  for (int i = 0; i < size; i++) cof_ptr[i] = i / 1000.0;

  float_sw4 *d_cof_ptr;
  d_cof_ptr = (float_sw4*) sycl::malloc_device(size * sizeof(float_sw4), q);
  q.memcpy(d_cof_ptr, cof_ptr, size * sizeof(float_sw4)).wait();

  // obtain memory offsets
  float_sw4 *d_sbop = d_cof_ptr;
  float_sw4 *d_acof = d_sbop + 6;
  float_sw4 *d_bop = d_acof + 384;
  float_sw4 *d_bope = d_bop + 24;
  float_sw4 *d_ghcof = d_bope + 48;
  float_sw4 *d_acof_no_gp = d_ghcof + 6;
  float_sw4 *d_ghcof_no_gp = d_acof_no_gp + 384;

  // Expected norm values after executing five kernels for the two input dataset
  float_sw4 exact_norm[2] = {2.2502232733796421194, 202.0512747393526638};

  for (int i = 0; i < 2; i++) {
    int* optr = onesided[i];
    float_sw4* alpha_ptr = arrays[i]["a_AlphaVE_0"]->m_data;
    size = arrays[i]["a_AlphaVE_0"]->m_nc *
           arrays[i]["a_AlphaVE_0"]->m_ni *
           arrays[i]["a_AlphaVE_0"]->m_nj *
           arrays[i]["a_AlphaVE_0"]->m_nk * sizeof(float_sw4);
    float_sw4* d_alpha_ptr;
    d_alpha_ptr = (float_sw4 *)sycl::malloc_device(size, q);
    q.memcpy(d_alpha_ptr, alpha_ptr, size);

    float_sw4* mua_ptr = arrays[i]["mMuVE_0"]->m_data;
    size = arrays[i]["mMuVE_0"]->m_nc *
           arrays[i]["mMuVE_0"]->m_ni *
           arrays[i]["mMuVE_0"]->m_nj *
           arrays[i]["mMuVE_0"]->m_nk * sizeof(float_sw4);
    float_sw4* d_mua_ptr;
    d_mua_ptr = (float_sw4 *)sycl::malloc_device(size, q);
    q.memcpy(d_mua_ptr, mua_ptr, size);

    float_sw4* lambdaa_ptr = arrays[i]["mLambdaVE_0"]->m_data;
    size = arrays[i]["mLambdaVE_0"]->m_nc *
           arrays[i]["mLambdaVE_0"]->m_ni *
           arrays[i]["mLambdaVE_0"]->m_nj *
           arrays[i]["mLambdaVE_0"]->m_nk * sizeof(float_sw4);
    float_sw4* d_lambdaa_ptr;
    d_lambdaa_ptr = (float_sw4 *)sycl::malloc_device(size, q);
    q.memcpy(d_lambdaa_ptr, lambdaa_ptr, size);

    float_sw4* met_ptr = arrays[i]["mMetric"]->m_data;
    size = arrays[i]["mMetric"]->m_nc *
           arrays[i]["mMetric"]->m_ni *
           arrays[i]["mMetric"]->m_nj *
           arrays[i]["mMetric"]->m_nk * sizeof(float_sw4);
    float_sw4* d_met_ptr;
    d_met_ptr = (float_sw4 *)sycl::malloc_device(size, q);
    q.memcpy(d_met_ptr, met_ptr, size);

    float_sw4* jac_ptr = arrays[i]["mJ"]->m_data;
    size = arrays[i]["mJ"]->m_nc *
           arrays[i]["mJ"]->m_ni *
           arrays[i]["mJ"]->m_nj *
           arrays[i]["mJ"]->m_nk * sizeof(float_sw4);
    float_sw4* d_jac_ptr;
    d_jac_ptr = (float_sw4 *)sycl::malloc_device(size, q);
    q.memcpy(d_jac_ptr, jac_ptr, size);

    float_sw4* uacc_ptr = arrays[i]["a_Uacc"]->m_data;
    // will initialize uacc content for each kernel run
    int uacc_size = arrays[i]["a_Uacc"]->m_nc *
                    arrays[i]["a_Uacc"]->m_ni *
                    arrays[i]["a_Uacc"]->m_nj *
                    arrays[i]["a_Uacc"]->m_nk * sizeof(float_sw4);
    float_sw4* d_uacc_ptr;
    d_uacc_ptr = (float_sw4 *) sycl::malloc_device(uacc_size, q);

    int* onesided_ptr = optr;
    int nkg = optr[12];
    char op = '-';

    int sg_str_size = (optr[7] - optr[6] + optr[9] - optr[8] + 2);
    float_sw4* sg_str = (float_sw4*) malloc (sg_str_size * sizeof(float_sw4));
    for (int n = 0; n < sg_str_size; n++) sg_str[n] = n / 1000.0;

    float_sw4* d_sg_str = sycl::malloc_device<float_sw4>(sg_str_size, q);
    q.memcpy(d_sg_str, sg_str, sg_str_size * sizeof(float_sw4));

    float_sw4* d_sg_str_x = d_sg_str;
    float_sw4* d_sg_str_y = d_sg_str_x + optr[7] - optr[6] + 1;

    double time = 0.0;

    // execute kernel (need to reset device uacc content for result verification)
    for (int p = 0; p < repeat; p++) {
      q.memcpy(d_uacc_ptr, uacc_ptr, uacc_size);

      q.wait();
      auto start = std::chrono::steady_clock::now();

      curvilinear4sg_ci(
          q,
          optr[6], optr[7], optr[8], optr[9], optr[10], optr[11],
          d_alpha_ptr, d_mua_ptr, d_lambdaa_ptr, d_met_ptr, d_jac_ptr,
          d_uacc_ptr, onesided_ptr, d_acof_no_gp, d_bope,
          d_ghcof_no_gp, d_acof_no_gp, d_ghcof_no_gp, d_sg_str_x,
          d_sg_str_y, nkg, op);

      q.wait();
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    std::cout << "\nAverage execution time of sw4ck kernels: "
              << (time * 1e-6f) / repeat << " milliseconds\n\n";

    size = arrays[i]["a_Uacc"]->m_nc *
           arrays[i]["a_Uacc"]->m_ni *
           arrays[i]["a_Uacc"]->m_nj *
           arrays[i]["a_Uacc"]->m_nk * sizeof(float_sw4);
    q.memcpy(uacc_ptr, d_uacc_ptr, size).wait();

    float_sw4 norm = arrays[i]["a_Uacc"]->norm();
    float_sw4 err = (norm - exact_norm[i]) / exact_norm[i] * 100;
    std::cout << "Error = " << err << " %\n";

    // Free host and device memory allocations
    sycl::free(d_alpha_ptr, q);
    sycl::free(d_mua_ptr, q);
    sycl::free(d_lambdaa_ptr, q);
    sycl::free(d_met_ptr, q);
    sycl::free(d_jac_ptr, q);
    sycl::free(d_uacc_ptr, q);
    sycl::free(d_sg_str, q);
    free(sg_str);
    delete(optr);
  }
  sycl::free(d_cof_ptr, q);
  free(cof_ptr);
  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i])
      delete(x.second);
  return 0;
}
