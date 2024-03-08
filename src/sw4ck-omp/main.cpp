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

#include "utils.h"
#include "utils.cpp"

int main(int argc, char* argv[]) {
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

  //
  // Allocate device memory explictly
  //
  int cof_size = (6 + 384 + 24 + 48 + 6 + 384 + 6 + 6);
  float_sw4 *cof_ptr = (float_sw4*) malloc (sizeof(float_sw4) * cof_size);
  for (int i = 0; i < cof_size; i++) cof_ptr[i] = i / 1000.0;

  #pragma omp target enter data map(to: cof_ptr[0:cof_size])

  /* obtain memory offsets
  float_sw4 *d_sbop = cof_ptr;
  float_sw4 *d_acof = sbop + 6;
  float_sw4 *d_bop = acof + 384;
  float_sw4 *d_bope = bop + 24;
  float_sw4 *d_ghcof = bope + 48;
  float_sw4 *d_acof_no_gp = ghcof + 6;
  float_sw4 *d_ghcof_no_gp = acof_no_gp + 384;
*/

  // Expected norm values after executing five kernels for the two input dataset
  float_sw4 exact_norm[2] = {2.2502232733796421194, 202.0512747393526638}; 

  for (int i = 0; i < 2; i++) {
    int* optr = onesided[i];
    float_sw4* alpha_ptr = arrays[i]["a_AlphaVE_0"]->m_data;
    int alpha_size = arrays[i]["a_AlphaVE_0"]->m_nc * 
           arrays[i]["a_AlphaVE_0"]->m_ni * 
           arrays[i]["a_AlphaVE_0"]->m_nj * 
           arrays[i]["a_AlphaVE_0"]->m_nk;

    float_sw4* mua_ptr = arrays[i]["mMuVE_0"]->m_data;
    int mua_size = arrays[i]["mMuVE_0"]->m_nc * 
           arrays[i]["mMuVE_0"]->m_ni * 
           arrays[i]["mMuVE_0"]->m_nj * 
           arrays[i]["mMuVE_0"]->m_nk;

    float_sw4* lambda_ptr = arrays[i]["mLambdaVE_0"]->m_data;
    int lambda_size = arrays[i]["mLambdaVE_0"]->m_nc * 
           arrays[i]["mLambdaVE_0"]->m_ni * 
           arrays[i]["mLambdaVE_0"]->m_nj * 
           arrays[i]["mLambdaVE_0"]->m_nk;

    float_sw4* met_ptr = arrays[i]["mMetric"]->m_data;
    int met_size = arrays[i]["mMetric"]->m_nc * 
           arrays[i]["mMetric"]->m_ni * 
           arrays[i]["mMetric"]->m_nj * 
           arrays[i]["mMetric"]->m_nk;

    float_sw4* jac_ptr = arrays[i]["mJ"]->m_data;
    int jac_size = arrays[i]["mJ"]->m_nc * 
           arrays[i]["mJ"]->m_ni * 
           arrays[i]["mJ"]->m_nj * 
           arrays[i]["mJ"]->m_nk;

    float_sw4* uacc_ptr = arrays[i]["a_Uacc"]->m_data;
    // will initialize uacc content for each kernel run
    int uacc_size = arrays[i]["a_Uacc"]->m_nc * 
                    arrays[i]["a_Uacc"]->m_ni * 
                    arrays[i]["a_Uacc"]->m_nj * 
                    arrays[i]["a_Uacc"]->m_nk;

    int* onesided_ptr = optr;
    int nkg = optr[12];
    char op = '-';

    int sg_str_size = (optr[7] - optr[6] + optr[9] - optr[8] + 2);
    float_sw4* sg_str = (float_sw4*) malloc (sg_str_size * sizeof(float_sw4));
    for (int n = 0; n < sg_str_size; n++) sg_str[n] = n / 1000.0; 

    //float_sw4* d_sg_str_y = d_sg_str_x + optr[7] - optr[6] + 1;

    // execute kernel (need to reset device uacc content for result verification)
    #pragma omp target data map(to: cof_ptr[0:cof_size], \
                               alpha_ptr[0:alpha_size], \
                               mua_ptr[0:mua_size], \
                               lambda_ptr[0:lambda_size], \
                               met_ptr[0:met_size],\
                               jac_ptr[0:jac_size], \
                               sg_str[0:sg_str_size]) \
                            map(alloc: uacc_ptr[0:uacc_size])
    {
      double time = 0.0;

      for (int p = 0; p < repeat; p++) {
        #pragma omp target update to (uacc_ptr[0:uacc_size])
        auto start = std::chrono::high_resolution_clock::now();

        curvilinear4sg_ci(optr[6], optr[7], optr[8], optr[9], optr[10], optr[11],
            alpha_ptr, mua_ptr, lambda_ptr, met_ptr, jac_ptr,
            uacc_ptr, onesided_ptr, cof_ptr, sg_str, nkg, op);

        auto end = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }

      std::cout << "\nAverage execution time of sw4ck kernels: "
                << (time * 1e-6f) / repeat << " milliseconds\n\n";

      #pragma omp target update from (uacc_ptr[0:uacc_size])
    }

    // Display the norms in hex and decimal formats before verification
    float_sw4 norm = arrays[i]["a_Uacc"]->norm();
    float_sw4 err = (norm - exact_norm[i]) / exact_norm[i] * 100;
    std::cout << "Error = " << err << " %\n";

    free(sg_str);
    delete(optr);
  }

  #pragma omp target exit data map(delete: cof_ptr[0:cof_size])
  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i]) 
      delete(x.second);
  return 0;
}
