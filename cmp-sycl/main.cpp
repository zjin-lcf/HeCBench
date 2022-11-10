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
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include "common.h"

#include "log.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "su_gather.hpp"

void
init_c(nd_item<1> &item, real *c, real inc, real c0) 
{
  int i = item.get_group(0);
  c[i] = c0 + inc*i;
}

void
init_half(nd_item<1> &item,
          const real* __restrict scalco, 
          const real* __restrict gx, 
          const real* __restrict gy, 
          const real* __restrict sx, 
          const real* __restrict sy, 
          real* __restrict h) 
{
  int i = item.get_group(0);
  real _s = scalco[i];

  if(-EPSILON < _s && _s < EPSILON) _s = 1.0f;
  else if(_s < 0) _s = 1.0f / _s;

  real hx = (gx[i] - sx[i]) * _s;
  real hy = (gy[i] - sy[i]) * _s;

  h[i] = 0.25f * (hx * hx + hy * hy) / FACTOR;
}

void
compute_semblances(nd_item<1> &item,
                   const real* __restrict h, 
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
  real _den = 0.0f, _ac_linear = 0.0f, _ac_squared = 0.0f;
  real _num[MAX_W],  m = 0.0f;
  int err = 0;

  int i = item.get_global_id(0);

  int t0 = i / nc;
  int c_id = i % nc;

  if(i < ns * nc)
  {
    real _c = c[c_id];
    real _t0 = _dt * t0;
    _t0 *= _t0;

    for(int j=0; j < _w; j++) _num[j] = 0.0f;

    for(int t_id=t_id0; t_id < t_idf; t_id++) {
      real t = sycl::sqrt(_t0 + _c * h[t_id]) * _idt;

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
redux_semblances(nd_item<1> &item,
                 const real* __restrict num, 
                 const real* __restrict stt, 
                 int*  __restrict ctr, 
                 real* __restrict str, 
                 real* __restrict stk,
                 const int nc, 
                 const int cdp_id,
                 const int ns) 
{
  int t0 = item.get_global_id(0);

  if(t0 < ns)
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

  real *h_gx, *h_gy, *h_sx, *h_sy, *h_scalco, *h_samples, *h_str, *h_stk, dt;
  int *ntraces_by_cdp_id, *h_ctr;

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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // Chronometer
  auto beg = std::chrono::high_resolution_clock::now();

  // Alloc memory
  buffer<real, 1> d_gx (h_gx, ttraces);
  buffer<real, 1> d_gy (h_gy, ttraces);
  buffer<real, 1> d_sx (h_sx, ttraces);
  buffer<real, 1> d_sy (h_sy, ttraces);
  buffer<real, 1> d_scalco (h_scalco, ttraces);
  buffer<real, 1> d_cdpsmpl (ntrs*ns);

  buffer<real, 1> d_c   (nc      );
  buffer<real, 1> d_h   (ttraces );
  buffer<real, 1> d_num (ns*nc   );
  buffer<real, 1> d_stt (ns*nc   );
  buffer< int, 1> d_ctr (ncdps*ns);
  buffer<real, 1> d_str (ncdps*ns);
  buffer<real, 1> d_stk (ncdps*ns);

  h_ctr = (int*) malloc (sizeof(int )*ncdps*ns);
  h_str = (real*) malloc (sizeof(real)*ncdps*ns);
  h_stk = (real*) malloc (sizeof(real)*ncdps*ns);

  //
  // DEVICE REGION
  //

  auto kbeg = std::chrono::high_resolution_clock::now();

  // Evaluate Cs - linspace
  q.submit([&] (handler &cgh) {
    auto c = d_c.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class k1>(nd_range<1>(range<1>(nc), range<1>(1)), [=] (nd_item<1> item) {
      init_c(item, c.get_pointer(), inc, c0);
    });
  });

  // Evaluate halfoffset points in x and y coordinates
  q.submit([&] (handler &cgh) {
    auto sc = d_scalco.get_access<sycl_read>(cgh);
    auto gx = d_gx.get_access<sycl_read>(cgh);
    auto gy = d_gy.get_access<sycl_read>(cgh);
    auto sx = d_sx.get_access<sycl_read>(cgh);
    auto sy = d_sy.get_access<sycl_read>(cgh);
    auto h = d_h.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class k2>(nd_range<1>(range<1>(ttraces), range<1>(1)), [=] (nd_item<1> item) {
      init_half(item, sc.get_pointer(), gx.get_pointer(), gy.get_pointer(),
                sx.get_pointer(), sy.get_pointer(), h.get_pointer());
    });
  });

  for(int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
    int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp_id];
    int stride = t_idf - t_id0;

    q.submit([&] (handler &cgh) {
      auto acc = d_cdpsmpl.get_access<sycl_discard_write>(cgh, stride*ns);
      cgh.copy(h_samples + t_id0*ns, acc);
    });

    // Compute semblances for each c for each sample
    q.submit([&] (handler &cgh) {
      auto h = d_h.get_access<sycl_read>(cgh);
      auto c = d_c.get_access<sycl_read>(cgh);
      auto smpl = d_cdpsmpl.get_access<sycl_read>(cgh);
      auto num = d_num.get_access<sycl_discard_write>(cgh);
      auto stt = d_stt.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class k3>(nd_range<1>(range<1>((ns*nc+NTHREADS-1)/NTHREADS*NTHREADS), 
                                             range<1>(NTHREADS)), [=] (nd_item<1> item) {
        compute_semblances(item, h.get_pointer(), c.get_pointer(), 
                           smpl.get_pointer(), num.get_pointer(), stt.get_pointer(), 
                           t_id0, t_idf, idt, dt, tau, w, nc, ns);
      });
    });

    // Get max C for max semblance for each sample on this cdp
    q.submit([&] (handler &cgh) {
      auto num = d_num.get_access<sycl_read>(cgh);
      auto stt = d_stt.get_access<sycl_read>(cgh);
      auto ctr = d_ctr.get_access<sycl_discard_write>(cgh);
      auto str = d_str.get_access<sycl_discard_write>(cgh);
      auto stk = d_stk.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class k4>(nd_range<1>(range<1>((ns+NTHREADS-1)/NTHREADS*NTHREADS), 
                                             range<1>(NTHREADS)), [=] (nd_item<1> item) {
        redux_semblances(item, num.get_pointer(), stt.get_pointer(), ctr.get_pointer(), 
                         str.get_pointer(), stk.get_pointer(), nc, cdp_id, ns);
      });
    });

    number_of_semblances += stride;

#ifdef DEBUG
    std::cout << "Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps) << std::endl;
#endif
  }
  // Gets time at end of computation
  q.wait();
  auto kend = std::chrono::high_resolution_clock::now();

  // Copy results back to host
  q.submit([&] (handler &cgh) {
    auto acc = d_ctr.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_ctr);
  });
  q.submit([&] (handler &cgh) {
    auto acc = d_str.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_str);
  });
  q.submit([&] (handler &cgh) {
    auto acc = d_stk.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_stk);
  });
  q.wait();

  auto end = std::chrono::high_resolution_clock::now();

  //
  // END DEVICE REGION
  //

  // Verify
  real* h_c   = (real*) malloc (sizeof(real)*nc      );
  real* h_h   = (real*) malloc (sizeof(real)*ttraces );
  real* h_num = (real*) malloc (sizeof(real)*ns*nc   );
  real* h_stt = (real*) malloc (sizeof(real)*ns*nc   );

  // reference results
   int* r_ctr = (int*) malloc (sizeof(int )*ncdps*ns);
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
   if (r_ctr[i] != h_ctr[i]) err_ctr++;
   if (r_str[i] - h_str[i] > 1e-3) err_str++;
   if (r_stk[i] - h_stk[i] > 1e-3) err_stk++;
  }
  float err_ctr_rate = (float)err_ctr / (ncdps * ns);
  float err_str_rate = (float)err_str / (ncdps * ns); 
  float err_stk_rate = (float)err_stk / (ncdps * ns); 
  printf("Error rate: ctr=%e str=%e stk=%e\n",
         err_ctr_rate, err_str_rate, err_stk_rate);

  // Logs stats (exec time and semblance-traces per second)
  double ktime = std::chrono::duration_cast<std::chrono::duration<double>>(kend - kbeg).count();
  double stps = (number_of_semblances / 1e9 ) * (ns * nc / ktime);
  std::string stats = "Giga-Semblances-Trace/s: " + std::to_string(stps);

  double offload_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
  stats += "\nDevice offload time: " + std::to_string(offload_time) + " (s) ";
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

    for(int k=0; k < ns; k++) ctr_t.data()[k] = h_ctr[i*ns+k] < 0 ? 0.0f: (c0 + inc * h_ctr[i*ns+k]) / FACTOR;
    str_t.data().assign(h_str + i*ns, h_str + (i+1)*ns);
    stk_t.data().assign(h_stk + i*ns, h_stk + (i+1)*ns);

    ctr_t.fputtr(c_out);
    str_t.fputtr(s_out);
    stk_t.fputtr(stack);
  }
#endif

  free(h_ctr    );
  free(h_str    );
  free(h_stk    );
  free(h_h      );
  free(h_c      );
  free(h_num    );
  free(h_stt    );
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
