////////////////////////////////////////////////////////////////////////////////
/**
 * @file main.cpp
 * @date 2017-03-04
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////


#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <omp.h>

#include "log.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "su_gather.hpp"

void
compute_semblances(const real* __restrict h, 
                   const real* __restrict c, 
                   const real* __restrict samples, 
                   real* __restrict num,
                   real* __restrict stt,
                   int t_id0, 
                   int t_idf,
                   real _idt,
                   real _dt,
                   int _tau,
                   int _w,
                   int nc,
                   int ns) 
{
  #pragma omp target teams distribute parallel for thread_limit(NTHREADS)
  for (int i = 0; i< ns*nc; i++) {

    real _den = 0.0f, _ac_linear = 0.0f, _ac_squared = 0.0f;
    real _num[MAX_W],  m = 0.0f;
    int err = 0;

    int t0 = i / nc;
    int c_id = i % nc;

    real _c = c[c_id];
    real _t0 = _dt * t0;
    _t0 *= _t0;

    for(int j=0; j < _w; j++) _num[j] = 0.0f;

    for(int t_id=t_id0; t_id < t_idf; t_id++) {
      real t = sqrtf(_t0 + _c * h[t_id]) * _idt;

      int it = (int)( t );
      int ittau = it - _tau;
      real x = t - (real)it;

      if(ittau >= 0 && it + _tau + 1 < ns) {
        int k1 = ittau + (t_id-t_id0)*ns;
        real sk1p1 = samples[k1], sk1;

        for(int j=0; j < _w; j++) {
          k1++;
          sk1 = sk1p1;
          sk1p1 = samples[k1];
          // linear interpolation optmized for this problema
          real v = (sk1p1 - sk1) * x + sk1;

          _num[j] += v;
          _den += v * v;
          _ac_linear += v;
        }
        m += 1;
      } else { err++; }
    }

    // Reduction for num
    for(int j=0; j < _w; j++) _ac_squared += _num[j] * _num[j];

    // Evaluate semblances
    if(_den > EPSILON && m > EPSILON && _w > EPSILON && err < 2) {
      num[i] = _ac_squared / (_den * m);
      stt[i] = _ac_linear  / (_w   * m);
    }
    else {
      num[i] = -1.0f;
      stt[i] = -1.0f;
    }
  }
}

void
redux_semblances(const real* __restrict num, 
                 const real* __restrict stt, 
                 int*  __restrict ctr, 
                 real* __restrict str, 
                 real* __restrict stk,
                 const int nc, 
                 const int cdp_id,
                 const int ns) 
{
  #pragma omp target teams distribute parallel for thread_limit(NTHREADS)
  for(int t0 = 0; t0 < ns; t0++)
  {
    real max_sem = 0.0f;
    int max_c = -1;

    for(int it=t0*nc; it < (t0+1)*nc ; it++) {
      real _num = num[it];
      if(_num > max_sem) {
        max_sem = _num;
        max_c = it;
      }
    }

    ctr[cdp_id*ns + t0] = max_c % nc;
    str[cdp_id*ns + t0] = max_sem;
    stk[cdp_id*ns + t0] = max_c > -1 ? stt[max_c] : 0;
  }
}

int main(int argc, const char** argv) {
#ifdef SAVE
  std::ofstream c_out("cmp.c.su", std::ofstream::out | std::ios::binary);
  std::ofstream s_out("cmp.coher.su", std::ofstream::out | std::ios::binary);
  std::ofstream stack("cmp.stack.su", std::ofstream::out | std::ios::binary);
#endif

  // Parse command line and read arguments
  parser::add_argument("-c0", "C0 constant");
  parser::add_argument("-c1", "C1 constant");
  parser::add_argument("-nc", "NC constant");
  parser::add_argument("-aph", "APH constant");
  parser::add_argument("-tau", "Tau constant");
  parser::add_argument("-i", "Data path");
  parser::add_argument("-v", "Verbosity Level 0-3");

  parser::parse(argc, argv);

  // Read parameters and input
  const real c0 = std::stof(parser::get("-c0", true)) * FACTOR;
  const real c1 = std::stof(parser::get("-c1", true)) * FACTOR;
  const real itau = std::stof(parser::get("-tau", true));
  const int nc = std::stoi(parser::get("-nc", true));
  const int aph = std::stoi(parser::get("-aph", true));
  std::string path = parser::get("-i", true);
  logger::verbosity_level(std::stoi(parser::get("-v", false)));

  // Reads *.su data and starts gather
  su_gather gather(path, aph, nc);

  real *h_gx, *h_gy, *h_sx, *h_sy, *h_scalco, *h_samples, dt;
  int *ntraces_by_cdp_id;

  // Linearize gather data in order to improve data coalescence in GPU
  gather.linearize(ntraces_by_cdp_id, h_samples, dt, h_gx, h_gy, h_sx, h_sy, h_scalco, nc);
  const int  ttraces = gather.ttraces(); // Total traces -> Total amount of traces read
  const int  ncdps = gather().size();    // Number of cdps -> Total number of cdps read
  const int  ns = gather.ns();           // Number of samples
  const int  ntrs = gather.ntrs();       // Max number of traces by cdp
  const real inc = (c1-c0) * (1.0f / (real)nc);


  dt = dt / 1000000.0f;
  real idt = 1.0f / dt;
  int tau = ((int)( itau * idt) > 0) ? ((int)( itau * idt)) : 0;
  int w = (2 * tau) + 1;

  int number_of_semblances = 0;

  LOG(INFO, "Starting CMP execution");

  // Alloc memory
  real *d_gx = h_gx;
  real *d_gy = h_gy;
  real *d_sx = h_sx;
  real *d_sy = h_sy;
  real *d_scalco = h_scalco;
  real *d_samples = h_samples;

  real *d_c   = (real*) malloc (sizeof(real) * nc      );
  real *d_h   = (real*) malloc (sizeof(real) * ttraces );
  real *d_num = (real*) malloc (sizeof(real) * ns*nc   );
  real *d_stt = (real*) malloc (sizeof(real) * ns*nc   );
   int *d_ctr = ( int*) malloc (sizeof(int) * ncdps*ns);
  real *d_str = (real*) malloc (sizeof(real) * ncdps*ns);
  real *d_stk = (real*) malloc (sizeof(real) * ncdps*ns);

  std::chrono::high_resolution_clock::time_point beg, end;
  //
  // DEVICE REGION
  //
  #pragma omp target data map(to: \
                                  d_gx[0:ttraces], \
                                  d_gy[0:ttraces], \
                                  d_sx[0:ttraces], \
                                  d_sy[0:ttraces], \
                                  d_scalco[0:ttraces],\
                                  d_samples[0:ttraces*ns])\
                          map(alloc:\
                                  d_c[0:nc], \
                                  d_h[0:ttraces], \
                                  d_num[0:ns*nc], \
                                  d_stt[0:ns*nc])\
                          map(from:\
                                  d_ctr[0:ncdps*ns], \
                                  d_str[0:ncdps*ns], \
                                  d_stk[0:ncdps*ns])
  {
    // Chronometer
    beg = std::chrono::high_resolution_clock::now();

    // Evaluate Cs - linspace
    #pragma omp target teams distribute parallel for thread_limit(1)
    for (int i = 0; i < nc; i++) 
      d_c[i] = c0 + inc*i;

    // Evaluate halfoffset points in x and y coordinates
    #pragma omp target teams distribute parallel for thread_limit(1)
    for (int i = 0; i < ttraces; i++) {
      real _s = d_scalco[i];

      if(-EPSILON < _s && _s < EPSILON) _s = 1.0f;
      else if(_s < 0) _s = 1.0f / _s;

      real hx = (d_gx[i] - d_sx[i]) * _s;
      real hy = (d_gy[i] - d_sy[i]) * _s;

      d_h[i] = 0.25f * (hx * hx + hy * hy) / FACTOR;
    }

    for(int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
      int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
      int t_idf = ntraces_by_cdp_id[cdp_id];
      int stride = t_idf - t_id0;

      // Compute semblances for each c for each sample
      compute_semblances(d_h, d_c, d_samples + t_id0*ns, d_num, d_stt,
                         t_id0, t_idf, idt, dt, tau, w, nc, ns);

      // Get max C for max semblance for each sample on this cdp
      redux_semblances(d_num, d_stt, d_ctr, d_str, d_stk, nc, cdp_id, ns);

      number_of_semblances += stride;

      #ifdef DEBUG
      std::cout << "Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps) << std::endl;
      #endif
    }

    // Gets time at end of computation
    end = std::chrono::high_resolution_clock::now();
  }

  //
  // END DEVICE REGION
  //

  // Verify
  real* h_c   = (real*) malloc (sizeof(real)*nc      );
  real* h_h   = (real*) malloc (sizeof(real)*ttraces );
  real* h_num = (real*) malloc (sizeof(real)*ns*nc   );
  real* h_stt = (real*) malloc (sizeof(real)*ns*nc   );

  // reference results
   int* r_ctr = ( int*) malloc (sizeof(int )*ncdps*ns);
  real* r_str = (real*) malloc (sizeof(real)*ncdps*ns);
  real* r_stk = (real*) malloc (sizeof(real)*ncdps*ns);

  h_init_c(nc, h_c, inc, c0);

  h_init_half(ttraces, h_scalco, h_gx, h_gy, h_sx, h_sy, h_h);

  for(int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
    int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp_id];

    // Compute semblances for each c for each sample
    h_compute_semblances(
        h_h, h_c, h_samples+t_id0*ns, h_num, h_stt, t_id0, t_idf, idt, dt, tau, w, nc, ns);

    // Get max C for max semblance for each sample on this cdp
    h_redux_semblances(h_num, h_stt, r_ctr, r_str, r_stk, nc, cdp_id, ns);
#ifdef DEBUG
    std::cout << "Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps) << std::endl;
#endif
  }

  int err_ctr = 0, err_str = 0, err_stk = 0;
  for (int i = 0; i < ncdps*ns; i++) {
   if (r_ctr[i] != d_ctr[i]) err_ctr++;
   if (r_str[i] - d_str[i] > 1e-3) err_str++;
   if (r_stk[i] - d_stk[i] > 1e-3) err_stk++;
  }
  float err_ctr_rate = (float)err_ctr / (ncdps * ns);
  float err_str_rate = (float)err_str / (ncdps * ns); 
  float err_stk_rate = (float)err_stk / (ncdps * ns); 
  printf("Error rate: ctr=%e str=%e stk=%e\n",
         err_ctr_rate, err_str_rate, err_stk_rate);

  // Logs stats (exec time and semblance-traces per second)
  double time = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
  double stps = (number_of_semblances / 1e9 ) * (ns * nc / time);
  std::string stats = "Giga semblances traces per second: " + std::to_string(stps);
  LOG(INFO, stats);

#ifdef SAVE
  // Delinearizes data and save it into a *.su file
  for(int i=0; i < ncdps; i++) {
    su_trace ctr_t = gather[i].traces()[0];
    su_trace str_t = gather[i].traces()[0];
    su_trace stk_t = gather[i].traces()[0];

    ctr_t.offset() = 0;
    ctr_t.sx() = ctr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
    ctr_t.sy() = ctr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;

    for(int k=0; k < ns; k++) ctr_t.data()[k] = d_ctr[i*ns+k] < 0 ? 0.0f: (c0 + inc * d_ctr[i*ns+k]) / FACTOR;
    str_t.data().assign(d_str + i*ns, d_str + (i+1)*ns);
    stk_t.data().assign(d_stk + i*ns, d_stk + (i+1)*ns);

    ctr_t.fputtr(c_out);
    str_t.fputtr(s_out);
    stk_t.fputtr(stack);
  }
#endif

  free(h_h      );
  free(d_h      );
  free(h_c      );
  free(d_c      );
  free(h_num    );
  free(d_num    );
  free(h_stt    );
  free(d_stt    );
  free(d_ctr    );
  free(d_str    );
  free(d_stk    );
  free(r_ctr    );
  free(r_str    );
  free(r_stk    );

  delete [] h_gx              ;
  delete [] h_gy              ;
  delete [] h_sx              ;
  delete [] h_sy              ;
  delete [] h_scalco          ;
  delete [] h_samples         ;
  delete [] ntraces_by_cdp_id ;

  return EXIT_SUCCESS;
}
