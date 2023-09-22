
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include "timer/timer.hpp"
#include "divergence.hpp"

constexpr const int DIMS = 2;

template <typename real>
void readVelocity(real *v, const int np, std::istream *input) {
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < np; j++) {
      for(int k = 0; k < np; k++) {
        (*input) >> v[k*np*2+2*j+i];
      }
    }
  }
}

template <int np, typename real>
void readElement(element<np, real> &elem,
                 std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> elem.metdet[i*np+j];
      elem.rmetdet[i*np+j] = 1 / elem.metdet[i*np+j];
    }
  }
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
      for(int k = 0; k < np; k++) {
        for(int l = 0; l < np; l++) {
          (*input) >> elem.Dinv[4*np*l+4*k+2*i+j];
        }
      }
    }
  }
}

template <int np, typename real>
void readDerivative(derivative<np, real> &deriv,
                    std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> deriv.Dvv[j*np+i];
    }
  }
}

template <typename real>
void readDivergence(real *divergence, const int np,
                    std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> divergence[i*np+j];
    }
  }
}


template <int np, typename real>
void compareDivergences(const real *v,
                        const element<np, real> &elem,
                        const derivative<np, real> &deriv,
                        const real *divergence_e,
                        const int numtests) {
  Timer::Timer time_c;
  /* Initial run to prevent cache timing from affecting us
   */
  std::cout << "Divergence on the CPU\n";
  real divergence_c[np*np];
  // warmup 
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, real>(v, deriv, elem, divergence_c);
  }

  time_c.startTimer();
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, real>(v, deriv, elem, divergence_c);
  }
  time_c.stopTimer();

  std::cout << "Divergence on the GPU\n";
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Timer::Timer time_f;
  real divergence_f[np*np];
  // warmup 
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(q, v, deriv, elem, divergence_f);
  }
  time_f.startTimer();
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(q, v, deriv, elem, divergence_f);
  }
  time_f.stopTimer();
  std::cout << "Divergence Errors\n";
  std::cout << "CPU             GPU\n";
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      std::cout << divergence_c[i*np+j] - divergence_e[i*np+j]
                << "    "
                << divergence_f[i*np+j] - divergence_e[i*np+j]
                << "\n";
    }
    std::cout << "\n";
  }

  std::cout << "CPU Time:\n" << time_c
            << "\n\nGPU Time:\n" << time_f << "\n";
}

int main(int argc, char **argv) {
  constexpr const int NP = 4;
  real v[NP*NP*DIMS];
  element<NP, real> elem;
  derivative<NP, real> deriv;
  real divergence_e[NP*NP];
  {
    std::istream *input;
    if(argc > 1) {
      input = new std::ifstream(argv[1]);
    } else {
      input = &std::cin;
    }
    readVelocity(v, NP, input);
    readElement(elem, input);
    readDerivative(deriv, input);
    readDivergence(divergence_e, NP, input);
    if(argc > 1) {
      delete input;
    }
  }

  constexpr const int defNumTests = 1e5;
  const int numtests = (argc > 2) ? std::stoi(argv[2]) : defNumTests;
  compareDivergences(v, elem, deriv, divergence_e, numtests);
  return 0;
}
